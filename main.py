import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import sys, io, os
import math
from typing import Dict, Any, Tuple, Optional

from  compression_chat_session import ChatSession

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--local_dir", default="./qwen2-7b-instruct")
    parser.add_argument("--mem_len", type=int, default=512)
    parser.add_argument("--compress_strides", type=str, default="4,4")
    parser.add_argument("--level_caps", type=str, default="1024,1024")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=4*1024)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--debug_compression", action="store_true", help="打印压缩检查日志与统计")
    parser.add_argument("--debug_interval", type=int, default=128, help="每多少新token打印一次压缩统计快照")
    parser.add_argument("--compress_mode", type=str, default="avg", choices=["avg", "mlp"],
                        help="压缩模式: avg=平均池化(无需训练), mlp=可训练MLP(需要load_weights)")
    parser.add_argument("--compress_layers", type=str, default="4",
                        help="压缩层配置: 数字N=后N层(如4=后4层,16=后16层), all=全部层, 或指定层号如'24,25,26,27'")
    
    # 训练相关参数
    parser.add_argument("--train", action="store_true", help="启用训练模式（自生成文本，无需外部数据集）")
    parser.add_argument("--train_samples", type=int, default=100, help="每个epoch训练样本数")
    parser.add_argument("--train_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--lambda_attn", type=float, default=1.0, help="注意力损失权重")
    parser.add_argument("--generation_length", type=int, default=8192, help="每个样本生成的token数（需要>mem_len才能触发压缩，建议8192-12288）")
    parser.add_argument("--load_weights", type=str, default=None, help="加载预训练的压缩器权重路径")
    
    args = parser.parse_args()

    strides = tuple(map(int, args.compress_strides.split(",")))
    caps = tuple(map(int, args.level_caps.split(",")))
    
    # 解析压缩层配置
    if args.compress_layers == "all":
        compress_layers = "all"
        compress_layers_desc = "全部层"
    elif args.compress_layers.isdigit():
        # 数字表示后N层,如 "4" 表示后4层, "16" 表示后16层
        n = int(args.compress_layers)
        compress_layers = n  # 直接传数字
        compress_layers_desc = f"后{n}层"
    elif "," in args.compress_layers:
        # 自定义层号，如 "24,25,26,27"
        compress_layers = list(map(int, args.compress_layers.split(",")))
        compress_layers_desc = f"指定层 {compress_layers}"
    else:
        raise ValueError(f"无效的 compress_layers 参数: {args.compress_layers}")

    session = ChatSession(
        model_name=args.model,
        local_dir=args.local_dir,
        mem_len=args.mem_len,
        compress_strides=strides,
        level_caps=caps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        debug_compression=args.debug_compression,
        debug_interval=args.debug_interval,
        compress_mode=args.compress_mode,
        compress_layers=compress_layers
    )
    
    # 打印压缩配置信息
    print(f"\n{'='*60}")
    print(f"压缩模式: {args.compress_mode.upper()}")
    if args.compress_mode == 'avg':
        print("  使用平均池化，无需训练权重")
    else:
        print("  使用MLP压缩器，需要训练权重")
    print(f"压缩层范围: {compress_layers_desc}")
    print(f"{'='*60}\n")
    
    # 加载预训练权重
    if args.load_weights:
        if args.compress_mode == 'avg':
            print("⚠️  警告: 平均池化模式不需要加载权重，将忽略 --load_weights 参数")
        else:
            session.load_compressor_weights(args.load_weights)
    elif args.compress_mode == 'mlp':
        print("⚠️  警告: MLP模式建议使用 --load_weights 加载训练好的权重，否则使用随机初始化权重")
    
    # 训练模式
    if args.train:
        print("\n[Training Mode] 自生成训练模式")
        print("[Info] 无需外部数据集，将使用模型自己生成的文本进行训练")
        print(f"[Info] 训练配置:")
        print(f"  - 每epoch样本数: {args.train_samples}")
        print(f"  - Epochs: {args.train_epochs}")
        print(f"  - Learning Rate: {args.learning_rate}")
        print(f"  - Lambda_Attn: {args.lambda_attn}")
        print(f"  - Generation Length: {args.generation_length} tokens")
        print(f"  - Mem Length: {args.mem_len} tokens")
        print(f"  - Compress Layers: 后4层 (layer 24-27)")
        
        session.train_compressor(
            num_samples=args.train_samples,
            num_epochs=args.train_epochs,
            learning_rate=args.learning_rate,
            lambda_attn=args.lambda_attn,
            generation_length=args.generation_length
        )
        
        print("\n训练完成！现在可以使用训练好的压缩器进行对话。")
        print("权重已保存到: compressor_weights.pt")

    print("已进入多轮对话模式。输入 /reset, /exit 可操作。")
    while True:
        try:
            s = input("\nuser> ").strip()
        except EOFError:
            print("\n[EOF] 再见。")
            break
        if not s:
            continue
        if s in {"/quit", "/exit"}:
            print("再见。")
            break
        if s == "/reset":
            session.reset()
            continue
        session.chat_once(s)


if __name__ == "__main__":
    main()

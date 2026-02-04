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

from module_compression import MultiLevelCompressiveKVCache

# ========== ModelScope 下载辅助 ==========
def download_model_with_modelscope(model_name="Qwen/Qwen2-7B-Instruct", local_dir="./qwen2-7b-instruct"):
    try:
        from modelscope import snapshot_download
        print("开始使用 ModelScope 下载模型...")
        model_dir = snapshot_download(model_name, cache_dir=local_dir, revision="master")
        print(f"模型已下载到: {model_dir}")
        return model_dir
    except ImportError:
        print("请先安装 ModelScope: pip install modelscope")
        raise
    except Exception as e:
        print(f"模型下载失败: {e}")
        raise


def _check_local_model_complete(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    must = ["config.json", "tokenizer.json"]
    for f in must:
        if not os.path.exists(os.path.join(local_dir, f)):
            return False
    ok = any([
        os.path.exists(os.path.join(local_dir, "pytorch_model.bin")),
        os.path.exists(os.path.join(local_dir, "model.safetensors")),
        any(n.startswith("pytorch_model-") and n.endswith(".bin") for n in os.listdir(local_dir)),
        any(n.startswith("model-") and n.endswith(".safetensors") for n in os.listdir(local_dir)),
    ])
    return ok

class ChatSession:
    def __init__(self, model_name, local_dir, mem_len=512,
                 compress_strides=(4, 4), level_caps=(1024, 1024),
                 temperature=0.8, top_p=0.95, max_new_tokens=4*1024, min_new_tokens=300, stop_on_eos=True,
                 debug_compression=False, debug_interval=128, compress_mode="avg", compress_layers=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.dtype = torch.float16 if self.use_cuda else torch.float32
        self.compress_mode = compress_mode  # 保存压缩模式
        self.compress_layers = compress_layers  # 保存压缩层配置

        model_path = local_dir if _check_local_model_complete(local_dir) else download_model_with_modelscope(model_name, local_dir)
        print(f"使用模型目录: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map={"": 0} if self.use_cuda else None,
            trust_remote_code=True
        ).to(self.device)
        if hasattr(self.model.config, "use_flash_attn"):
            self.model.config.use_flash_attn = False
        self.model.eval()
        torch.set_grad_enabled(False)

        self.messages = []
        self.seq_len_tracker = 0
        self.ckvc = None
        self.past = None
        self.n_layers = None

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.stop_on_eos = stop_on_eos
        self.mem_len = mem_len
        self.compress_strides = compress_strides
        self.level_caps = level_caps
        self.eos_id = getattr(self.model.config, "eos_token_id", None)
        if isinstance(self.eos_id, (list, tuple)) and len(self.eos_id) > 0:
            self.eos_id = self.eos_id[0]

        # 调试与检查
        self.debug_compression = debug_compression
        self.debug_interval = max(1, int(debug_interval))

        # 控制压缩回写频率（可选优化）
        self._steps_since_rebuild = 0
        self._rebuild_every = 16  # 保险起见，最多每 16 步强制重建一次

    def _build_full_ids(self):
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        text = ""
        for m in self.messages:
            role = m["role"]
            prefix = "User" if role == "user" else "Assistant"
            text += f"{prefix}: {m['content']}\n"
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    def _prefill(self, new_full_ids):
        with torch.inference_mode():
            if self.past is None:
                out = self.model(input_ids=new_full_ids, use_cache=True, return_dict=True)
                self.past = out.past_key_values
                self.n_layers = len(self.past.key_cache) if hasattr(self.past, "key_cache") else len(self.past)
                
                # ★ 修复：只在CKVC不存在时创建（避免覆盖已加载的权重）
                if self.ckvc is None:
                    
                    # 获取模型配置
                    d_model = self.model.config.hidden_size
                    num_heads = self.model.config.num_attention_heads
                    num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
                    
                    self.ckvc = MultiLevelCompressiveKVCache(
                        n_layers=self.n_layers,
                        compress_strides=self.compress_strides,
                        mem_len=self.mem_len,
                        level_caps=self.level_caps,
                        compress_mode=self.compress_mode,  # 使用传入的压缩模式
                        d_model=d_model,
                        num_heads=num_heads,
                        num_key_value_heads=num_kv_heads,  # GQA支持
                        use_attention_loss=True,
                        compress_layers=self.compress_layers,  # 使用传入的压缩层配置
                        enable_debug=self.debug_compression
                    ).to(self.device, dtype=torch.float32)  # 显式指定FP32保证稳定性
                
                # 初始化：灌入 CKVC，并构建一次压缩后的 past
                self.ckvc.update_from_model_past_init(self.past)
                self.past = self.ckvc.build_compressed_past()
                
                # ★ 关键：首次prefill后也要同步prev_total_len（否则下一步delta会错）
                self.ckvc.sync_prev_total_len_from_current_past(self.past)
                
                self.seq_len_tracker = new_full_ids.shape[1]
                self._steps_since_rebuild = 0

                if self.debug_compression:
                    print("[DEBUG] CKVC stats after initial prefill:")
                    print(self.ckvc.get_stats())

                return new_full_ids[:, -1:]
            delta = new_full_ids.shape[1] - self.seq_len_tracker
            delta_ids = new_full_ids[:, -delta:]
            
            if self.debug_compression:
                if self.past is not None:
                    past_lens = [self.past.key_cache[i].shape[2] if hasattr(self.past, 'key_cache') else self.past[i][0].shape[2] for i in range(len(self.past.key_cache if hasattr(self.past, 'key_cache') else self.past))]
                    print(f"[DEBUG] Before model call: past_lens(first 5)={past_lens[:5]}, past_lens(last 5)={past_lens[-5:]}, delta_ids={delta_ids.shape[1]}")
                else:
                    print(f"[DEBUG] Before model call: past=None, delta_ids={delta_ids.shape[1]}")
            
            out = self.model(input_ids=delta_ids, use_cache=True, past_key_values=self.past, return_dict=True)
            
            if self.debug_compression:
                ret_past_lens = [out.past_key_values.key_cache[i].shape[2] if hasattr(out.past_key_values, 'key_cache') else out.past_key_values[i][0].shape[2] for i in range(len(out.past_key_values.key_cache if hasattr(out.past_key_values, 'key_cache') else out.past_key_values))]
                print(f"[DEBUG] After model call: returned_past_lens(first 5)={ret_past_lens[:5]}, returned_past_lens(last 5)={ret_past_lens[-5:]}")

            # 增量 prefill：更新 CKVC 跟踪
            changed = self.ckvc.update_from_model_past_infer(out.past_key_values)
            # 关键：每次都重建past，因为CKVC内部状态已更新
            self.past = self.ckvc.build_compressed_past_infer()
            
            # 更新prev_total_len以反映新构建的past的实际长度（用于下次计算delta）
            self.ckvc.sync_prev_total_len_from_current_past(self.past)
            self._steps_since_rebuild = 0

            self.seq_len_tracker = new_full_ids.shape[1]

            if self.debug_compression:
                print("[DEBUG] CKVC stats after incremental prefill:")
                print(self.ckvc.get_stats())

            return delta_ids[:, -1:]

    def _sample_next_id(self, logits):
        probs = torch.softmax(logits / max(1e-6, self.temperature), dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum <= self.top_p
            mask[..., 0] = True
            filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(filtered, num_samples=1)
            next_id = sorted_idx.gather(-1, idx)
        else:
            next_id = torch.multinomial(probs, num_samples=1)
        return next_id

    def _maybe_debug_tick(self, new_tokens):
        if self.debug_compression and (new_tokens % self.debug_interval == 0):
            stats = self.ckvc.get_stats()
            print(f"\n[DEBUG] Step {new_tokens} compression stats snapshot:\n{stats}\n")
    
    def get_compression_stats(self):
        """ 获取当前压缩统计信息（用于Web界面可视化）
        包含以下信息：
           1.原始区token数量（即mem_len_now)
           2.l1压缩后token数量(即l1_len_now)
           3.l2压缩后token数量(即l2_len_now)
           4.压缩事件次数
           5.压缩前后token总数
        """
        if self.ckvc is None:
            return None
        
        # 计算各层平均值（只统计compress_layers）
        mem_tokens = 0
        l1_tokens = 0
        l2_tokens = 0
        compress_layer_count = len(self.ckvc.compress_layers)
        
        for layer_idx in self.ckvc.compress_layers:
            mem_len_now = 0 if self.ckvc.mem_k[layer_idx] is None else self.ckvc.mem_k[layer_idx].shape[2]
            l1_len_now = 0 if self.ckvc.level_k[0][layer_idx] is None else self.ckvc.level_k[0][layer_idx].shape[2]
            l2_len_now = 0 if (self.ckvc.num_levels < 2 or self.ckvc.level_k[1][layer_idx] is None) else self.ckvc.level_k[1][layer_idx].shape[2]
            
            mem_tokens += mem_len_now
            l1_tokens += l1_len_now
            l2_tokens += l2_len_now
        
        # 计算平均值
        if compress_layer_count > 0:
            mem_tokens = mem_tokens // compress_layer_count
            l1_tokens = l1_tokens // compress_layer_count
            l2_tokens = l2_tokens // compress_layer_count
        
        # 获取压缩事件次数
        stats = self.ckvc.get_stats()
        l1_compress_events = stats.get('l1_compress_events', 0)
        l2_compress_events = stats.get('l2_compress_events', 0)
        
        # 计算压缩前后token总数
        total_tokens = mem_tokens + l1_tokens + l2_tokens
        # 压缩前的估算：还原压缩
        stride1 = self.ckvc.compress_strides[0]
        stride2 = self.ckvc.compress_strides[1] if len(self.ckvc.compress_strides) > 1 else 1
        original_tokens = mem_tokens + l1_tokens * stride1 + l2_tokens * stride1 * stride2
        
        return {
            'mem_tokens': mem_tokens,
            'l1_tokens': l1_tokens,
            'l2_tokens': l2_tokens,
            'mem_cap': self.ckvc.mem_len,
            'l1_cap': self.ckvc.level_caps[0],
            'l2_cap': self.ckvc.level_caps[1] if len(self.ckvc.level_caps) > 1 else 0,
            'l1_compress_events': l1_compress_events,
            'l2_compress_events': l2_compress_events,
            'total_tokens': total_tokens,
            'original_tokens': original_tokens,
            'compress_strides': self.ckvc.compress_strides,
        }
    
    def generate_stream(self):
        full_ids = self._build_full_ids()
        pieces = []

        token_buffer = []
        buffer_size = 8  # 稍大一点，防止断字节

        def _emit(ids_tensor=None, is_final=False):
            if ids_tensor is not None:
                if ids_tensor.dim() == 1:
                    ids_tensor = ids_tensor.unsqueeze(0)
                token_buffer.append(ids_tensor)

            if not token_buffer:
                return

            # 到达缓冲阈值或最终输出时 decode
            if is_final or len(token_buffer) >= buffer_size:
                combined_tokens = torch.cat(token_buffer, dim=1)
                s = self.tokenizer.decode(
                    combined_tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    errors='ignore'
                )
                print(s, end="", flush=True)
                pieces.append(s)
                token_buffer.clear()

        with torch.inference_mode():
            last_token = self._prefill(full_ids)
            new_tokens = 0

            while new_tokens < self.max_new_tokens:
                out = self.model(
                    input_ids=last_token,
                    use_cache=True,
                    past_key_values=self.past,
                    return_dict=True
                )

                next_logits = out.logits[:, -1, :]

                # 防止提前停止
                if self.eos_id is not None and new_tokens < self.min_new_tokens:
                    next_logits[:, self.eos_id] = -float("inf")

                next_id = self._sample_next_id(next_logits)
                if next_id.dim() == 1:
                    next_id = next_id.unsqueeze(0)

                token_int = int(next_id.item())
                new_tokens += 1

                _emit(next_id)

                # EOS 检查
                if self.stop_on_eos and self.eos_id is not None and new_tokens >= self.min_new_tokens:
                    if token_int == int(self.eos_id):
                        break

                last_token = next_id

                # 关键：让 CKVC 跟踪，并在发生变化或到达周期时重建 past
                changed = self.ckvc.update_from_model_past_infer(out.past_key_values)
                self._steps_since_rebuild += 1
                if changed : #or (self._steps_since_rebuild >= self._rebuild_every)
                    self.past = self.ckvc.build_compressed_past_infer()
                    # 压缩past：prev_total_len跟压缩后的长度对齐
                    self.ckvc.sync_prev_total_len_from_current_past(self.past)
                    self._steps_since_rebuild = 0
                else:
                    # 不压缩：沿用HF返回的原始past，但prev_total_len也要跟它对齐
                    self.past = out.past_key_values
                    self.ckvc.sync_prev_total_len_from_current_past(self.past)

                self._maybe_debug_tick(new_tokens)

            # 处理剩余 token
            _emit(is_final=True)

        if self.debug_compression:
            print("\n[DEBUG] Final compression stats:")
            print(self.ckvc.get_stats())

        return "".join(pieces)

    def chat_once(self, text):
        self.messages.append({"role": "user", "content": text})
        print("\nassistant> ", end="", flush=True)
        reply = self.generate_stream()
        print("")
        self.messages.append({"role": "assistant", "content": reply})

    def reset(self):
        self.messages = []
        self.seq_len_tracker = 0
        self.ckvc = None
        self.past = None
        self.n_layers = None
        self._steps_since_rebuild = 0
    
    def train_compressor(self, num_samples=100, num_epochs=3, learning_rate=1e-4, 
                        lambda_attn=1.0, generation_length=512):
        """
        训练压缩器（使用自生成文本，无需外部数据集）
        
        原理：
        1. 用多样化的提示词让模型生成长文本
        2. 生成过程中自动触发KV cache压缩
        3. 计算压缩前后attention输出的差异作为损失
        4. 只训练压缩器参数，模型参数冻结
        
        参数:
            num_samples: 每个epoch生成多少个样本
            num_epochs: 训练轮数
            learning_rate: 学习率
            lambda_attn: 注意力损失权重
            generation_length: 每个样本生成的token数（需要>mem_len才能触发压缩）
        """
        print("\n[Training] 开始训练压缩器（自生成模式）...")
        print(f"[Info] 配置:")
        print(f"  - 每epoch样本数: {num_samples}")
        print(f"  - 总epoch数: {num_epochs}")
        print(f"  - 学习率: {learning_rate}")
        print(f"  - 生成长度: {generation_length} tokens (mem_len={self.mem_len})")
        print(f"  - 压缩触发: 当生成>{self.mem_len}个token时自动压缩")
        
        # 确保CKVC已初始化
        if self.ckvc is None:
            print("[Debug] 初始化CKVC...")
            # 初始化一个dummy past来创建CKVC
            dummy_ids = self.tokenizer("初始化", return_tensors="pt")["input_ids"].to(self.device)
            print("[Debug] dummy_ids创建完成")
            with torch.no_grad():
                print("[Debug] 开始模型forward...")
                out = self.model(input_ids=dummy_ids, use_cache=True, return_dict=True)
                print("[Debug] 模型forward完成")
                self.past = out.past_key_values
                self.n_layers = len(self.past.key_cache)
                
                print(f"[Debug] n_layers={self.n_layers}")
                d_model = self.model.config.hidden_size
                num_heads = self.model.config.num_attention_heads
                num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
                print(f"[Debug] d_model={d_model}, num_heads={num_heads}, num_kv_heads={num_kv_heads}")
                
                print("[Debug] 创建CKVC对象...")
                self.ckvc = MultiLevelCompressiveKVCache(
                    n_layers=self.n_layers,
                    compress_strides=self.compress_strides,
                    mem_len=self.mem_len,
                    level_caps=self.level_caps,
                    compress_mode=self.compress_mode,  # 使用传入的压缩模式
                    d_model=d_model,
                    num_heads=num_heads,
                    num_key_value_heads=num_kv_heads,
                    use_attention_loss=True,
                    compress_layers=self.compress_layers,  # 使用传入的压缩层配置
                    enable_debug=False
                ).to(self.device, dtype=torch.float32)  # 显式FP32，训练稳定
                print("[Debug] CKVC对象创建完成")
                
                # 训练时使用FP32压缩器（数值稳定），推理时可以用FP16
                print(f"[Info] Compressor dtype: {next(self.ckvc.parameters()).dtype} (FP32 for stability)")
                print(f"[Info] Model dtype: {next(self.model.parameters()).dtype}")
        
        print("[Debug] 设置模型为eval模式并冻结参数...")
        # 冻结基座模型，只训练压缩器（大幅节省显存）
        self.model.eval()  # eval模式
        self.model.requires_grad_(False)  # 批量冻结所有参数
        print("[Debug] 模型参数冻结完成")
        
        # 确保CKVC的参数可训练
        self.ckvc.train()
        self.ckvc.requires_grad_(True)
        
        # 启用梯度计算（但只针对CKVC）
        torch.set_grad_enabled(True)
        print("[Debug] 梯度计算已启用")
        
        # 优化器（只优化CKVC的参数）
        trainable_params = list(self.ckvc.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        print(f"[Debug] 优化器创建完成，可训练参数: {sum(p.numel() for p in trainable_params)}")
        
        # 多样化的提示词（用于生成训练样本）- 使用更能引导长文本的提示
        prompts = [
            "写一篇关于人工智能发展历史的详细文章，包括各个时代的重要突破、代表人物、技术细节和社会影响。从图灵测试讲到深度学习，每个阶段都要详细展开。",
            "详细介绍中国所有主要传统节日和习俗，包括春节、元宵节、清明节、端午节、中秋节、重阳节等。每个节日都要介绍起源、庆祝方式、地域差异和现代演变。",
            "全面解释量子力学的基本原理和应用，从波粒二象性、不确定性原理讲到量子纠缠、量子隧穿和量子计算。包括历史发展、数学基础和实际应用。",
            "深入讨论气候变化对地球生态系统的影响，分析温室效应的机制、极端天气事件、生物多样性丧失、海平面上升、冰川融化等各个方面，并探讨应对策略。",
            "全面分析莎士比亚作品的文学价值，包括四大悲剧、四大喜剧的主题、人物塑造、语言艺术、戏剧结构，以及对后世文学的影响。",
            "详细介绍编程语言Python的特点和用途，包括语法特性、标准库、第三方生态、在数据科学、机器学习、Web开发等领域的应用，以及与其他语言的比较。",
            "探讨未来城市发展的趋势，包括智慧城市技术、可持续建筑、公共交通革新、绿色能源应用、城市规划理念、社会治理创新等多个维度。",
            "讲述中国古代四大发明的详细故事，包括造纸术、印刷术、火药、指南针的发明过程、技术演进、传播路径和对世界文明的深远影响。",
            "比较东西方饮食文化的差异，从食材选择、烹饪方式、用餐礼仪、餐桌文化、节日食品到饮食哲学，全面分析两种文化体系的特点。",
            "解释区块链技术的工作原理，包括分布式账本、共识机制、密码学基础、智能合约、去中心化应用，以及在金融、供应链等领域的应用前景。",
            "描述太阳系各大行星的详细特征，包括水星、金星、地球、火星、木星、土星、天王星、海王星的物理性质、大气组成、卫星系统和探测历史。",
            "讨论教育改革的必要性和方向，分析传统教育的问题、素质教育的理念、个性化学习、技术辅助教学、教育公平等关键议题。",
            "介绍世界著名建筑奇迹的历史和特点，包括金字塔、长城、泰姬陵、埃菲尔铁塔、悉尼歌剧院等，讲述其建造背景、建筑技术和文化意义。",
            "分析经济全球化的利弊，探讨贸易自由化、资本流动、技术转移、文化交流的积极影响，以及贫富差距、环境问题、文化同质化等挑战。",
            "讲解深度学习的核心技术，包括神经网络结构、反向传播算法、卷积神经网络、循环神经网络、Transformer架构，以及在计算机视觉和自然语言处理中的应用。",
        ]
        
        print("[Debug] 开始训练循环...")
        try:
            for epoch in range(num_epochs):
                epoch_loss_attn = 0.0
                num_samples_processed = 0
                print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
                
                for idx in range(num_samples):
                    print(f"\n[Sample {idx+1}/{num_samples}] 开始生成训练样本...")
                    # 定期清理显存
                    if idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    # 随机选择提示词
                    prompt = prompts[idx % len(prompts)]
                    print(f"[Sample {idx+1}] 提示词: {prompt[:30]}...")
                    
                    # 维护当前样本的损失值（用于日志打印）
                    current_sample_loss = None
                    
                    # 重置 CKVC 的缓存状态
                    print(f"[Sample {idx+1}] 重置CKVC状态...")
                    for i in range(self.ckvc.n_layers):  # 修复: 使用self.ckvc.n_layers而不是self.n_layers
                        self.ckvc.mem_k[i] = None
                        self.ckvc.mem_v[i] = None
                        for level in range(self.ckvc.num_levels):
                            self.ckvc.level_k[level][i] = None
                            self.ckvc.level_v[level][i] = None
                        self.ckvc.res_l1_k[i] = None
                        self.ckvc.res_l1_v[i] = None
                        self.ckvc.res_l2_k[i] = None
                        self.ckvc.res_l2_v[i] = None
                        self.ckvc.prev_total_len[i] = 0
                    
                    self.ckvc.enable_training_mode()
                    self.ckvc.reset_accumulated_loss()
                    
                    # Tokenize提示词（不使用chat template，直接用原始文本）
                    input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                    
                    # 初始化：用提示词做prefill
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=input_ids,
                            use_cache=True,
                            return_dict=True
                        )
                        past_kv = outputs.past_key_values
                    
                    # 更新CKVC（第一次，使用init模式）
                    self.ckvc.update_from_model_past_init(past_kv)
                    
                    # 获取压缩后的past用于后续生成
                    model_past = self.ckvc.build_compressed_past_infer()
                    
                    # 关键：同步prev_total_len与压缩后的past对齐（训练路径也需要）
                    self.ckvc.sync_prev_total_len_from_current_past(model_past)
                    
                    # 生成文本（逐token，使用压缩后的KV cache）
                    with torch.enable_grad():
                        for step in range(generation_length):
                            # Forward（模型部分no_grad）
                            with torch.no_grad():
                                # 采样最后一个token作为输入
                                next_input = input_ids[:, -1:]
                                
                                outputs = self.model(
                                    input_ids=next_input,
                                    past_key_values=model_past,
                                    use_cache=True,
                                    return_dict=True
                                )
                                
                                logits = outputs.logits
                                new_past = outputs.past_key_values
                                
                                # 采样下一个token
                                next_token_logits = logits[:, -1, :]
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                                input_ids = torch.cat([input_ids, next_token], dim=1)
                            
                            # 更新CKVC并累积损失（有梯度）
                            changed = self.ckvc.update_from_model_past_infer(new_past)
                            
                            # 如果CKVC发生压缩，需要重建model_past并同步
                            if changed:
                                model_past = self.ckvc.build_compressed_past_infer()
                                # 关键：同步prev_total_len（训练路径也需要）
                                self.ckvc.sync_prev_total_len_from_current_past(model_past)
                        
                        # 获取累积的注意力损失（在生成循环后，仍在sample循环内）
                        attn_loss_dict = self.ckvc.get_accumulated_loss()
                        
                        # 打印压缩统计（每个样本）
                        if idx < 5 or (idx + 1) % 20 == 0:  # 前5个样本和每20个样本打印一次
                            current_stats = self.ckvc.get_stats()
                            print(f"  [Sample {idx+1}] 压缩统计:")
                            print(f"    L1_events={current_stats.get('l1_compress_events', 0)}, "
                                  f"L2_events={current_stats.get('l2_compress_events', 0)}")
                            if self.ckvc.compress_layers:
                                last_layer = max(self.ckvc.compress_layers)
                                print(f"    Layer{last_layer}: mem→L1={current_stats.get('compressed_from_mem_to_l1', [0]*self.ckvc.n_layers)[last_layer]}, "
                                      f"L1→L2={current_stats.get('compressed_from_l1_to_l2', [0]*self.ckvc.n_layers)[last_layer]}")
                        
                        if attn_loss_dict and isinstance(attn_loss_dict['total'], torch.Tensor):
                            total_loss = lambda_attn * attn_loss_dict['total']
                            
                            # Backward（必须在if内部，否则total_loss可能是None）
                            optimizer.zero_grad()
                            total_loss.backward()
                            
                            # 梯度裁剪（防止梯度爆炸）
                            torch.nn.utils.clip_grad_norm_(self.ckvc.parameters(), max_norm=1.0)
                            
                            optimizer.step()
                            
                            epoch_loss_attn += total_loss.item()
                            num_samples_processed += 1
                            
                            # 保存当前样本的损失值用于日志
                            current_sample_loss = total_loss.item()
                            
                            # 打印进度
                            if (idx + 1) % 10 == 0:
                                stats = self.ckvc.get_stats()
                                avg_loss = epoch_loss_attn / num_samples_processed if num_samples_processed > 0 else 0
                                print(f"  Epoch {epoch+1}/{num_epochs}, Sample {idx+1}/{num_samples}: "
                                      f"Loss_Attn={avg_loss:.4f}, "
                                      f"L1_compress={stats.get('l1_compress_events', 0)}, "
                                      f"L2_compress={stats.get('l2_compress_events', 0)}")
                        else:
                            # 如果没有累积到有效损失（例如生成长度不够触发压缩），跳过这个样本
                            print(f"  [Warning] Sample {idx+1}: No valid loss accumulated, skipping backward")
                        
                        # 每50个样本检查梯度
                        if (idx + 1) % 50 == 0:
                            grad_norm = 0.0
                            for param in self.ckvc.parameters():
                                if param.grad is not None:
                                    grad_norm += param.grad.norm().item() ** 2
                            grad_norm = grad_norm ** 0.5
                            # 使用current_sample_loss，如果为None则显示N/A
                            loss_str = f"{current_sample_loss:.4f}" if current_sample_loss is not None else "N/A"
                            print(f"  [Grad] Sample {idx+1}: grad_norm={grad_norm:.4f}, loss={loss_str}")
                
                # Epoch统计
                avg_loss_attn = epoch_loss_attn / num_samples_processed if num_samples_processed > 0 else 0
                final_stats = self.ckvc.get_stats()
                
                print(f"\nEpoch {epoch+1}/{num_epochs} 完成:")
                print(f"  Loss_Attn: {avg_loss_attn:.4f}")
                print(f"  压缩事件: L1={final_stats.get('l1_compress_events', 0)}, L2={final_stats.get('l2_compress_events', 0)}")
        
        except Exception as e:
            print(f"\n[Error] 训练过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n[Training] 训练完成！")
        
        # 恢复推理模式
        self.ckvc.disable_training_mode()
        torch.set_grad_enabled(False)
        
        # ★ 根据压缩层配置生成权重文件名
        if self.compress_layers == "all":
            weight_suffix = "all"
            config_desc = "全部28层"
        elif isinstance(self.compress_layers, int):
            weight_suffix = str(self.compress_layers)
            config_desc = f"后{self.compress_layers}层"
        elif isinstance(self.compress_layers, (list, set)):
            # 如果是指定层号列表,使用层数
            num_layers = len(self.compress_layers)
            weight_suffix = str(num_layers)
            layer_ids = sorted(self.compress_layers) if isinstance(self.compress_layers, set) else sorted(self.compress_layers)
            config_desc = f"{num_layers}层 (layers {layer_ids})"
        else:
            weight_suffix = "custom"
            config_desc = "自定义配置"
        
        save_path = f"compressor_weights_{weight_suffix}.pt"
        
        # 保存压缩器参数(包含配置信息)
        checkpoint = {
            'state_dict': self.ckvc.state_dict(),
            'config': {
                'compress_layers': self.compress_layers,
                'num_compress_layers': self.ckvc.num_compress_layers,
                'compress_layer_ids': sorted(self.ckvc.compress_layers),
                'compress_strides': self.compress_strides,
                'mem_len': self.mem_len,
                'level_caps': self.level_caps,
                'compress_mode': self.compress_mode
            }
        }
        torch.save(checkpoint, save_path)
        
        print(f"\n{'='*80}")
        print(f"[Success] 压缩器权重已保存")
        print(f"{'='*80}")
        print(f"保存路径: {save_path}")
        print(f"压缩配置: {config_desc}")
        print(f"压缩器数量: {self.ckvc.num_compress_layers}")
        print(f"压缩层ID: {sorted(self.ckvc.compress_layers)}")
        print(f"\n文件命名规则:")
        print(f"  - compressor_weights_4.pt   → 后4层 (Layer 24-27)")
        print(f"  - compressor_weights_8.pt   → 后8层 (Layer 20-27)")
        print(f"  - compressor_weights_16.pt  → 后16层 (Layer 12-27)")
        print(f"  - compressor_weights_all.pt → 全部28层")
        print(f"{'='*80}\n")
    
    def load_compressor_weights(self, weight_path="compressor_weights.pt"):
        """加载训练好的压缩器权重"""
        if not os.path.exists(weight_path):
            print(f"[Warning] 权重文件不存在: {weight_path}")
            return
        
        # 先加载到CPU检查格式
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # 检查是否是新格式(包含config)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config = checkpoint['config']
            state_dict_cpu = checkpoint['state_dict']
            
            print(f"\n{'='*80}")
            print(f"[Info] 权重文件配置信息:")
            print(f"{'='*80}")
            print(f"文件路径: {weight_path}")
            print(f"压缩层数: {config['num_compress_layers']}")
            print(f"压缩层ID: {config['compress_layer_ids']}")
            print(f"压缩模式: {config['compress_mode']}")
            print(f"压缩步长: {config['compress_strides']}")
            print(f"Mem长度: {config['mem_len']}")
            print(f"Level容量: {config['level_caps']}")
            
            # 验证当前配置是否匹配
            if hasattr(self, 'compress_layers'):
                expected_num = len(self.compress_layers) if isinstance(self.compress_layers, (list, set)) else \
                              (self.n_layers if self.compress_layers == "all" else self.compress_layers)
                if expected_num != config['num_compress_layers']:
                    print(f"\n⚠️  警告: 配置不匹配!")
                    print(f"  权重文件: {config['num_compress_layers']}层")
                    print(f"  当前配置: {expected_num}层")
                    print(f"  建议: 使用 --compress_layers {config['num_compress_layers']} 参数")
                    print(f"{'='*80}\n")
                    return
        else:
            # 旧格式(直接是state_dict)
            state_dict_cpu = checkpoint
            print(f"[Info] 加载旧格式权重文件(无配置信息): {weight_path}")
        
        # 如果CKVC未初始化，先初始化它
        if self.ckvc is None:
            print("[Info] 自动初始化CKVC以加载权重...")
            # 使用一个简短的dummy输入来初始化CKVC
            dummy_ids = self.tokenizer("初始化", return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=dummy_ids, use_cache=True, return_dict=True)
                self.past = out.past_key_values
                self.n_layers = len(self.past.key_cache) if hasattr(self.past, "key_cache") else len(self.past)
                
                # 获取模型配置
                d_model = self.model.config.hidden_size
                num_heads = self.model.config.num_attention_heads
                num_kv_heads = getattr(self.model.config, 'num_key_value_heads', num_heads)
                
                # 创建CKVC实例 - 使用self.compress_layers而不是None
                self.ckvc = MultiLevelCompressiveKVCache(
                    n_layers=self.n_layers,
                    compress_strides=self.compress_strides,
                    mem_len=self.mem_len,
                    level_caps=self.level_caps,
                    compress_mode="mlp",
                    d_model=d_model,
                    num_heads=num_heads,
                    num_key_value_heads=num_kv_heads,
                    use_attention_loss=True,
                    compress_layers=self.compress_layers,  # ★ 使用传入的compress_layers配置
                    enable_debug=self.debug_compression
                ).to(self.device, dtype=torch.float32)
                
                # 初始化CKVC状态
                self.ckvc.update_from_model_past_init(self.past)
                print(f"[Info] CKVC已自动初始化 (压缩层: {sorted(self.ckvc.compress_layers)})")
        
        # 加载权重到目标设备
        state_dict = {k: v.to(self.device) for k, v in state_dict_cpu.items()}
        self.ckvc.load_state_dict(state_dict)
        
        print(f"\n{'='*80}")
        print(f"[Success] 权重加载成功")
        print(f"{'='*80}")
        print(f"压缩器数量: {self.ckvc.num_compress_layers}")
        print(f"压缩层ID: {sorted(self.ckvc.compress_layers)}")
        print(f"权重设备: {next(self.ckvc.parameters()).device}")
        
        # 验证权重已加载并显示详细信息
        if hasattr(self.ckvc, 'compress_k_l1') and len(self.ckvc.compress_k_l1) > 0:
            print(f"\n权重验证:")
            # 显示前3个压缩器的权重
            for i in range(min(3, len(self.ckvc.compress_k_l1))):
                weights = self.ckvc.compress_k_l1[i].weight[0, 0, :].tolist()
                print(f"  压缩器 {i}: [{', '.join(f'{w:.6f}' for w in weights)}]")
            
            # 检查是否是初始值
            first_weight = self.ckvc.compress_k_l1[0].weight[0, 0, :].tolist()
            if all(abs(w - 0.25) < 1e-5 for w in first_weight):
                print(f"\n  ⚠️  警告: 权重为初始值 (0.25), 可能未训练!")
            else:
                deviation = max(abs(w - 0.25) for w in first_weight)
                print(f"\n  ✅ 权重已训练 (最大偏离初始值: {deviation:.6f})")
        
        print(f"{'='*80}\n")
        
        # 清理缓存并重置对话状态
        self.past = None
        self.messages = []
        torch.cuda.empty_cache()
        print("[Info] 已重置对话与缓存，权重加载完成")
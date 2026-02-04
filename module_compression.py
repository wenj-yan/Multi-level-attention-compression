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

class MultiLevelCompressiveKVCache(nn.Module):
    """
    多级记忆机制（无 sink）：
    - 最近 mem_len 个 token 保留原始分辨率；
    - 超出部分依次进行多级压缩（compress_strides 控制每级下采样率）
    - 使用可学习的MLP代替平均池化
    - 通过注意力分数对比损失优化压缩质量
    """

    def __init__(self, n_layers, compress_strides=(4, 4), mem_len=512, level_caps=(1024, 1024),
                 compress_mode="mlp", enable_debug=False, 
                 d_model=4096, num_heads=32, num_key_value_heads=None, use_attention_loss=True, compress_layers=None):
        super().__init__()
        
        self.n_layers = n_layers
        self.compress_strides = compress_strides
        self.level_caps = level_caps
        self.num_levels = len(compress_strides)
        self.compress_mode = compress_mode
        self.enable_debug = enable_debug
        
        # 压缩层配置：支持多种模式
        if compress_layers is None:
            self.compress_layers = set(range(max(0, n_layers - 4), n_layers))  # 默认后4层
        elif compress_layers == "all":
            self.compress_layers = set(range(n_layers))  # 全部层
        elif isinstance(compress_layers, int):
            # 数字N表示后N层
            self.compress_layers = set(range(max(0, n_layers - compress_layers), n_layers))
        else:
            self.compress_layers = set(compress_layers)  # 自定义层列表
        
        # 创建层号到压缩器索引的映射 (只为压缩层分配索引)
        self.layer_to_compressor_idx = {}
        for idx, layer_id in enumerate(sorted(self.compress_layers)):
            self.layer_to_compressor_idx[layer_id] = idx
        self.num_compress_layers = len(self.compress_layers)
        
        # 为每层设置独立的参数：非压缩层使用超大mem_len和stride=1（不压缩）
        self.mem_len_per_layer = []
        self.compress_strides_per_layer = []
        for i in range(n_layers):
            if i in self.compress_layers:
                # 压缩层：使用正常参数
                self.mem_len_per_layer.append(mem_len)
                self.compress_strides_per_layer.append(compress_strides)
            else:
                # 非压缩层：使用相同的mem_len，避免无限累积OOM
                # 超出部分会移到L1/L2但不压缩（完整保存）
                self.mem_len_per_layer.append(mem_len)
                self.compress_strides_per_layer.append((1, 1))  # stride=1表示不压缩
        
        # 保留全局参数供训练等场景使用
        self.mem_len = mem_len
        self.compress_strides = compress_strides
        self.level_caps = level_caps
        
        # 新增：模型参数
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_attention_loss = use_attention_loss
        
        # GQA支持：计算KV通道数
        if num_key_value_heads is None:
            num_key_value_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_channels = num_key_value_heads * self.head_dim
        
        # 可学习的压缩层（MLP/Conv）- 只为压缩层创建独立参数
        if compress_mode == "mlp":
            # L1压缩：为每个压缩层创建独立的压缩器
            self.compress_k_l1 = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.kv_channels,
                    out_channels=self.kv_channels,
                    kernel_size=compress_strides[0],
                    stride=compress_strides[0],
                    groups=self.kv_channels,  # Depthwise: 每个维度独立
                    bias=False
                ) for _ in range(self.num_compress_layers)  # 只创建压缩层数量的压缩器
            ])
            self.compress_v_l1 = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.kv_channels,
                    out_channels=self.kv_channels,
                    kernel_size=compress_strides[0],
                    stride=compress_strides[0],
                    groups=self.kv_channels,
                    bias=False
                ) for _ in range(self.num_compress_layers)
            ])
            
            # L2压缩：为每个压缩层创建独立的压缩器
            if len(compress_strides) > 1:
                self.compress_k_l2 = nn.ModuleList([
                    nn.Conv1d(
                        in_channels=self.kv_channels,
                        out_channels=self.kv_channels,
                        kernel_size=compress_strides[1],
                        stride=compress_strides[1],
                        groups=self.kv_channels,
                        bias=False
                    ) for _ in range(self.num_compress_layers)
                ])
                self.compress_v_l2 = nn.ModuleList([
                    nn.Conv1d(
                        in_channels=self.kv_channels,
                        out_channels=self.kv_channels,
                        kernel_size=compress_strides[1],
                        stride=compress_strides[1],
                        groups=self.kv_channels,
                        bias=False
                    ) for _ in range(self.num_compress_layers)
                ])
        
        # 初始化压缩层权重（防止NaN）- 只初始化压缩层
        if compress_mode == "mlp":
            # 使用平均池化初始化（在FP32下稳定）
            with torch.no_grad():
                stride1 = compress_strides[0]
                for comp_idx in range(self.num_compress_layers):
                    self.compress_k_l1[comp_idx].weight.fill_(1.0 / stride1)
                    self.compress_v_l1[comp_idx].weight.fill_(1.0 / stride1)
                if len(compress_strides) > 1:
                    stride2 = compress_strides[1]
                    for comp_idx in range(self.num_compress_layers):
                        self.compress_k_l2[comp_idx].weight.fill_(1.0 / stride2)
                        self.compress_v_l2[comp_idx].weight.fill_(1.0 / stride2)
        
        # 训练模式相关
        self.training_mode = False
        # compress_mode 在 __init__ 参数中传入，不要在这里硬编码
        self.accumulated_loss = {
            'total': 0.0,
            'attn_dist': 0.0,
            'attn_output': 0.0,
            'count': 0
        }

        self.mem_k = [None] * n_layers
        self.mem_v = [None] * n_layers
        self.level_k = [[None] * n_layers for _ in range(self.num_levels)]
        self.level_v = [[None] * n_layers for _ in range(self.num_levels)]

        # 残留缓冲：攒够 stride 再压缩
        self.res_l1_k = [None] * n_layers
        self.res_l1_v = [None] * n_layers
        self.res_l2_k = [None] * n_layers
        self.res_l2_v = [None] * n_layers

        self.device = None
        self.prev_total_len = [0] * n_layers
        self.compress_num = [0] * n_layers

        # 统计信息
        self.stats: Dict[str, Any] = {
            "layers": n_layers,
            "mem_len_cap": mem_len,
            "level_caps": list(level_caps),
            "compress_strides": list(compress_strides),
            "events": 0,
            "l1_compress_events": 0,
            "l2_compress_events": 0,
            "tokens_pushed_raw_per_layer": [0] * n_layers,     # 推入 mem 的原始 token 数
            "tokens_kept_mem_per_layer": [0] * n_layers,       # mem 当前 token 数
            "tokens_level1_per_layer": [0] * n_layers,         # L1 当前 token 数（压缩后）
            "tokens_level2_per_layer": [0] * n_layers,         # L2 当前 token 数（压缩后）
            "compressed_from_mem_to_l1": [0] * n_layers,       # 进入 L1 的压缩后 token 数累积
            "compressed_from_l1_to_l2": [0] * n_layers,        # 进入 L2 的压缩后 token 数累积
            "residual_l1_per_layer": [0] * n_layers,           # L1 残留区长度（未压缩）
            "residual_l2_per_layer": [0] * n_layers,           # L2 残留区长度（未压缩）
        }

    def reset_state(self):
        """重置所有缓存状态（保留配置和可学习参数）"""
        # 清空KV缓存
        self.mem_k = [None] * self.n_layers
        self.mem_v = [None] * self.n_layers
        self.level_k = [[None] * self.n_layers for _ in range(self.num_levels)]
        self.level_v = [[None] * self.n_layers for _ in range(self.num_levels)]
        
        # 清空残留缓冲
        self.res_l1_k = [None] * self.n_layers
        self.res_l1_v = [None] * self.n_layers
        self.res_l2_k = [None] * self.n_layers
        self.res_l2_v = [None] * self.n_layers
        
        # 重置计数器
        self.prev_total_len = [0] * self.n_layers
        self.compress_num = [0] * self.n_layers
        
        # 重置统计信息
        self.stats["events"] = 0
        self.stats["l1_compress_events"] = 0
        self.stats["l2_compress_events"] = 0
        self.stats["tokens_pushed_raw_per_layer"] = [0] * self.n_layers
        self.stats["tokens_kept_mem_per_layer"] = [0] * self.n_layers
        self.stats["tokens_level1_per_layer"] = [0] * self.n_layers
        self.stats["tokens_level2_per_layer"] = [0] * self.n_layers
        self.stats["compressed_from_mem_to_l1"] = [0] * self.n_layers
        self.stats["compressed_from_l1_to_l2"] = [0] * self.n_layers
        self.stats["residual_l1_per_layer"] = [0] * self.n_layers
        self.stats["residual_l2_per_layer"] = [0] * self.n_layers

    @staticmethod
    def _cat(a, b, dim=2):
        if a is None: return b
        if b is None: return a
        return torch.cat([a, b], dim=dim)

    @staticmethod
    def _take_prefix(x, n):
        if x is None or n <= 0: return None
        return x[:, :, :n, :].contiguous()

    @staticmethod
    def _take_suffix(x, n):
        if x is None or n <= 0: return x
        T = x.shape[2]
        if n >= T: return None
        return x[:, :, n:, :].contiguous()

    def _append(self, dst, src):
        return self._cat(dst, src, dim=2)

    def _compress_chunk(self, k_chunk, v_chunk, stride, level=0, query=None, layer_idx=0):
        """
        使用平均池化或可学习的MLP/Conv进行压缩
        
        参数:
            k_chunk, v_chunk: 待压缩的KV
            stride: 压缩率
            level: 压缩级别（0=L1, 1=L2）
            query: 当前query（用于计算注意力损失，训练时需要）
            layer_idx: 层索引（用于判断是否需要MLP压缩）
        
        返回:
            k_comp, v_comp, loss_dict
        """
        if k_chunk is None or v_chunk is None:
            return None, None, None
        
        bs, h, T, d = k_chunk.shape
        if T < stride:
            return None, None, None
        
        # stride=1表示不压缩，直接返回原始数据（用于非压缩层）
        if stride == 1:
            return k_chunk, v_chunk, None
        
        # 非压缩层：完全不压缩，直接返回None
        if layer_idx not in self.compress_layers:
            return None, None, None
        
        # === 以下仅对压缩层执行 ===
        
        # 模式1: 平均池化（快速、稳定、无需训练）
        if self.compress_mode == "avg":
            # 计算压缩后的长度
            T_comp = T // stride
            if T_comp == 0:
                return None, None, None
            
            # 重塑并平均池化：[bs, h, T, d] -> [bs, h, T_comp, stride, d] -> [bs, h, T_comp, d]
            k_reshaped = k_chunk[:, :, :T_comp*stride, :].reshape(bs, h, T_comp, stride, d)
            v_reshaped = v_chunk[:, :, :T_comp*stride, :].reshape(bs, h, T_comp, stride, d)
            
            k_comp = k_reshaped.mean(dim=3)  # [bs, h, T_comp, d]
            v_comp = v_reshaped.mean(dim=3)
            
            return k_comp, v_comp, None
        
        # 模式2: MLP压缩器（可训练，需要训练后的权重）
        if self.compress_mode != "mlp":
            raise ValueError(f"不支持的压缩模式: {self.compress_mode}，请使用 'avg' 或 'mlp'")
        
        # 保存原始KV（用于损失计算）
        k_original = k_chunk.clone() if self.training_mode and query is not None else None
        v_original = v_chunk.clone() if self.training_mode and query is not None else None
        
        # 压缩层：使用可学习的卷积压缩
        # 重塑: [bs, h_kv, T, d] -> [bs, h_kv*d, T]
        # 注意：h_kv是KV heads数量（GQA中可能<num_heads）
        k_t = k_chunk.permute(0, 1, 3, 2).contiguous()  # [bs, h_kv, d, T]
        v_t = v_chunk.permute(0, 1, 3, 2).contiguous()
        
        k_flat = k_t.reshape(bs, h * d, T)
        v_flat = v_t.reshape(bs, h * d, T)
        
        # 检查输入是否有NaN
        if torch.isnan(k_flat).any() or torch.isnan(v_flat).any():
            print(f"[CRITICAL] NaN detected in INPUT to compressor! level={level}, layer={layer_idx}")
            return None, None, None
        
        # FP16数值稳定性：裁剪输入范围（防止卷积溢出）
        input_dtype = k_flat.dtype
        # 获取当前层对应的压缩器索引
        comp_idx = self.layer_to_compressor_idx[layer_idx]
        
        if k_flat.dtype == torch.float16 or k_flat.dtype == torch.bfloat16:
            k_flat = torch.clamp(k_flat, -50.0, 50.0)
            v_flat = torch.clamp(v_flat, -50.0, 50.0)
            # 转换为FP32进行压缩（如果压缩器是FP32）
            if self.compress_k_l1[comp_idx].weight.dtype == torch.float32:
                k_flat = k_flat.float()
                v_flat = v_flat.float()
        
        # 选择压缩层 - 使用压缩器索引而不是层号
        if level == 0:
            # 检查权重是否有NaN
            if torch.isnan(self.compress_k_l1[comp_idx].weight).any():
                print(f"[CRITICAL] Compressor K L1 weights contain NaN at layer {layer_idx} (comp_idx {comp_idx})!")
                return None, None, None
            k_comp = self.compress_k_l1[comp_idx](k_flat)  # 使用压缩器索引
            v_comp = self.compress_v_l1[comp_idx](v_flat)
        elif level == 1 and hasattr(self, 'compress_k_l2'):
            k_comp = self.compress_k_l2[comp_idx](k_flat)  # 使用压缩器索引
            v_comp = self.compress_v_l2[comp_idx](v_flat)
        else:
            raise ValueError(f"Invalid compression level: {level}")
        
        # 转回原始dtype
        if input_dtype != k_comp.dtype:
            k_comp = k_comp.to(input_dtype)
            v_comp = v_comp.to(input_dtype)
        
        # 检查压缩输出是否有NaN
        if torch.isnan(k_comp).any() or torch.isnan(v_comp).any():
            print(f"[CRITICAL] NaN detected in compressor output! level={level}, layer={layer_idx}")
            print(f"  Input k range: [{k_flat.min().item():.4f}, {k_flat.max().item():.4f}], mean={k_flat.mean().item():.4f}")
            print(f"  Input v range: [{v_flat.min().item():.4f}, {v_flat.max().item():.4f}], mean={v_flat.mean().item():.4f}")
            print(f"  Output k range: [{k_comp.min().item():.4f}, {k_comp.max().item():.4f}]")
            print(f"  Output v range: [{v_comp.min().item():.4f}, {v_comp.max().item():.4f}]")
            print(f"  Input dtype: {k_flat.dtype}, Output dtype: {k_comp.dtype}")
            print(f"  Compressor weight dtype: {self.compress_k_l1[comp_idx].weight.dtype if level==0 else self.compress_k_l2[comp_idx].weight.dtype}")
            # 返回None避免传播NaN
            return None, None, None
        
        # 恢复形状: [bs, h*d, T_comp] -> [bs, h_kv, T_comp, d]
        T_comp = k_comp.shape[2]
        k_comp = k_comp.reshape(bs, h, d, T_comp).permute(0, 1, 3, 2).contiguous()
        v_comp = v_comp.reshape(bs, h, d, T_comp).permute(0, 1, 3, 2).contiguous()
        
        # 计算注意力损失（仅在训练模式且有query时）
        loss_dict = None
        if self.training_mode and query is not None and self.use_attention_loss:
            loss_dict = self._compute_attention_preservation_loss(
                query, k_original, k_comp, v_original, v_comp
            )
        
        return k_comp, v_comp, loss_dict

    def _truncate_keep_tail(self, x, keep_len):
        if x is None: return None
        tail = x.shape[2] - keep_len
        if tail <= 0: return x
        return x[:, :, -keep_len:, :].contiguous()
    
    def _compute_attention_scores(self, query, key):
        """
        计算注意力分数
        
        query: [batch, heads, 1, head_dim]
        key: [batch, heads, seq_len, head_dim]
        返回: [batch, heads, 1, seq_len]
        """
        # 缩放点积注意力
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        return attn_weights
    
    def _align_attention(self, attn_weights, stride):
        """
        将原始注意力分数按压缩比例聚合
        
        attn_weights: [batch, heads, 1, seq_len]
        stride: 压缩率
        返回: [batch, heads, 1, seq_len//stride]
        """
        batch, heads, _, seq_len = attn_weights.shape
        seq_len_compressed = seq_len // stride
        
        # 重塑并求和
        aligned = attn_weights.view(batch, heads, 1, seq_len_compressed, stride)
        aligned = aligned.sum(dim=-1)  # [batch, heads, 1, seq_len_compressed]
        
        # 重新归一化
        aligned = aligned / (aligned.sum(dim=-1, keepdim=True) + 1e-10)
        
        return aligned
    
    def _compute_attention_preservation_loss(self, query, k_original, k_compressed, 
                                            v_original, v_compressed):
        """
        计算注意力输出损失：对比使用完整KV和压缩KV时的attention输出差异
        
        这是正确的优化目标：让压缩后的attention输出尽可能接近完整KV的输出
        
        参数:
            query: [bs, h_q, 1, d] 查询向量（通常是最后一个token）
            k_original, v_original: 压缩前的KV [bs, h_kv, T_orig, d]
            k_compressed, v_compressed: 压缩后的KV [bs, h_kv, T_comp, d]
        
        返回:
            loss_dict: {'total': Tensor, 'attn_dist': float, 'attn_output': float}
        """
        if query is None:
            # 如果没有query（初始化阶段），回退到KV重建损失
            return self._compute_kv_reconstruction_loss(k_original, k_compressed, 
                                                       v_original, v_compressed)
        
        # 获取形状
        bs, h_kv, T_orig, d = k_original.shape
        _, h_q, q_len, _ = query.shape
        T_comp = k_compressed.shape[2]
        
        # 数值稳定性检查
        if torch.isnan(query).any() or torch.isnan(k_original).any() or torch.isnan(v_original).any():
            print(f"[WARNING] NaN in inputs before attention calculation")
            return None
        
        if torch.isnan(k_compressed).any() or torch.isnan(v_compressed).any():
            print(f"[WARNING] NaN in compressed KV before attention calculation")
            return None
        
        # 统一dtype到FP32进行loss计算（数值稳定）
        compute_dtype = torch.float32
        query = query.to(compute_dtype)
        k_original = k_original.to(compute_dtype)
        v_original = v_original.to(compute_dtype)
        k_compressed = k_compressed.to(compute_dtype)
        v_compressed = v_compressed.to(compute_dtype)
        
        # GQA: 扩展KV heads到匹配query heads
        # h_q = 28, h_kv = 4, 每个KV head对应 28/4=7 个query heads
        num_key_value_groups = h_q // h_kv
        
        # 扩展完整KV: [bs, h_kv, T, d] -> [bs, h_q, T, d]
        k_full_expanded = k_original.repeat_interleave(num_key_value_groups, dim=1)
        v_full_expanded = v_original.repeat_interleave(num_key_value_groups, dim=1)
        
        # 扩展压缩KV: [bs, h_kv, T_comp, d] -> [bs, h_q, T_comp, d]
        k_comp_expanded = k_compressed.repeat_interleave(num_key_value_groups, dim=1)
        v_comp_expanded = v_compressed.repeat_interleave(num_key_value_groups, dim=1)
        
        # 1. 计算使用完整KV的attention输出
        # Attention scores: [bs, h_q, 1, T_orig]
        scale = d ** -0.5
        attn_scores_full = torch.matmul(query, k_full_expanded.transpose(-2, -1)) * scale
        attn_weights_full = F.softmax(attn_scores_full, dim=-1)
        attn_output_full = torch.matmul(attn_weights_full, v_full_expanded)  # [bs, h_q, 1, d]
        
        # 2. 计算使用压缩KV的attention输出
        # Attention scores: [bs, h_q, 1, T_comp]
        attn_scores_comp = torch.matmul(query, k_comp_expanded.transpose(-2, -1)) * scale
        attn_weights_comp = F.softmax(attn_scores_comp, dim=-1)
        attn_output_comp = torch.matmul(attn_weights_comp, v_comp_expanded)  # [bs, h_q, 1, d]
        
        # 检查输出是否有NaN
        if torch.isnan(attn_output_full).any() or torch.isnan(attn_output_comp).any():
            print(f"[WARNING] NaN in attention outputs")
            return None
        
        # 3. 计算两个输出的差异
        # MSE损失（归一化）
        with torch.no_grad():
            output_std = attn_output_full.std() + 1e-6
        
        loss_attn_output = F.mse_loss(attn_output_comp, attn_output_full) / output_std
        
        # 余弦相似度损失（采样部分heads）
        loss_cos = 0.0
        num_sample_heads = min(h_q, 8)  # 采样8个heads
        for i in range(0, h_q, h_q // num_sample_heads):
            if i >= h_q:
                break
            full_i = attn_output_full[:, i, :, :].reshape(bs, -1)
            comp_i = attn_output_comp[:, i, :, :].reshape(bs, -1)
            loss_cos += (1.0 - F.cosine_similarity(full_i, comp_i, dim=1).mean())
        loss_cos = loss_cos / num_sample_heads
        
        # 组合损失
        total_loss = loss_attn_output + 0.5 * loss_cos
        total_loss = torch.clamp(total_loss, 0.0, 100.0)  # 防止梯度爆炸
        
        return {
            'total': total_loss,
            'attn_dist': 0.0,
            'attn_output': total_loss.item()
        }
    
    def _compute_kv_reconstruction_loss(self, k_original, k_compressed, v_original, v_compressed):
        """
        KV重建损失（备用方案，当没有query时使用）
        """
        bs, h, T_orig, d = k_original.shape
        T_comp = k_compressed.shape[2]
        stride = T_orig // T_comp
        
        # 上采样压缩的KV
        k_upsampled = k_compressed.repeat_interleave(stride, dim=2)
        v_upsampled = v_compressed.repeat_interleave(stride, dim=2)
        
        if k_upsampled.shape[2] != T_orig:
            k_upsampled = k_upsampled[:, :, :T_orig, :]
            v_upsampled = v_upsampled[:, :, :T_orig, :]
        
        # 计算MSE
        with torch.no_grad():
            k_std = k_original.std() + 1e-6
            v_std = v_original.std() + 1e-6
        
        loss_k = F.mse_loss(k_upsampled, k_original) / k_std
        loss_v = F.mse_loss(v_upsampled, v_original) / v_std
        
        total_loss = loss_k + loss_v
        total_loss = torch.clamp(total_loss, 0.0, 100.0)
        
        return {
            'total': total_loss,
            'attn_dist': 0.0,
            'attn_output': total_loss.item()
        }

    def _update_layer_lengths_stats(self, layer_idx):
        mem_len_now = 0 if self.mem_k[layer_idx] is None else self.mem_k[layer_idx].shape[2]
        l1_len_now = 0 if self.level_k[0][layer_idx] is None else self.level_k[0][layer_idx].shape[2]
        l2_len_now = 0 if (self.num_levels < 2 or self.level_k[1][layer_idx] is None) else self.level_k[1][layer_idx].shape[2]
        self.stats["tokens_kept_mem_per_layer"][layer_idx] = mem_len_now
        self.stats["tokens_level1_per_layer"][layer_idx] = l1_len_now
        if self.num_levels > 1:
            self.stats["tokens_level2_per_layer"][layer_idx] = l2_len_now
        # 残留统计
        r1 = 0 if self.res_l1_k[layer_idx] is None else self.res_l1_k[layer_idx].shape[2]
        r2 = 0 if self.res_l2_k[layer_idx] is None else self.res_l2_k[layer_idx].shape[2]
        self.stats["residual_l1_per_layer"][layer_idx] = r1
        self.stats["residual_l2_per_layer"][layer_idx] = r2

    def update_from_model_past_init(self, past_key_values) -> bool:
        """
        将模型返回的最新 past 增量喂入 CKVC，
        并根据 mem_len / caps / strides 进行批处理压缩。
        返回值: 是否发生了“影响past布局”的修改（用于决定是否需要重建 past）
        """
        changed = False
        if past_key_values is None:
            return changed

        if hasattr(past_key_values, "key_cache"):
            k_all_list = past_key_values.key_cache
            v_all_list = past_key_values.value_cache
        else:
            k_all_list = [kv[0] for kv in past_key_values]
            v_all_list = [kv[1] for kv in past_key_values]

        if self.device is None and k_all_list:
            self.device = k_all_list[0].device

        self.stats["events"] += 1

        for layer_idx in range(self.n_layers):
            k_all = k_all_list[layer_idx]
            v_all = v_all_list[layer_idx]
            total_seq = k_all.shape[2]
            
            # 计算增量
            delta = total_seq - self.prev_total_len[layer_idx]
            if delta <= 0:
                continue

            new_k = k_all[:, :, -delta:, :].contiguous()
            new_v = v_all[:, :, -delta:, :].contiguous()
            self.stats["tokens_pushed_raw_per_layer"][layer_idx] += int(delta)

            # 先拼到 mem
            self.mem_k[layer_idx] = self._append(self.mem_k[layer_idx], new_k)
            self.mem_v[layer_idx] = self._append(self.mem_v[layer_idx], new_v)

            # 一级压缩：使用各层独立的mem_len和strides
            layer_mem_len = self.mem_len_per_layer[layer_idx]
            layer_strides = self.compress_strides_per_layer[layer_idx]
            overflow = self.mem_k[layer_idx].shape[2] - layer_mem_len
            
            # 非压缩层：将溢出部分完整移到L1（不压缩）
            if layer_idx not in self.compress_layers:
                if overflow > 0:
                    # 将溢出的部分移到L1（不压缩，完整保存）
                    overflow_k = self.mem_k[layer_idx][:, :, :overflow, :]
                    overflow_v = self.mem_v[layer_idx][:, :, :overflow, :]
                    
                    # 添加到L1（使用level_k[0]，不是self.l1_k）
                    self.level_k[0][layer_idx] = self._append(self.level_k[0][layer_idx], overflow_k)
                    self.level_v[0][layer_idx] = self._append(self.level_v[0][layer_idx], overflow_v)
                    
                    # mem保留最新的部分
                    self.mem_k[layer_idx] = self._take_suffix(self.mem_k[layer_idx], overflow)
                    self.mem_v[layer_idx] = self._take_suffix(self.mem_v[layer_idx], overflow)
                    
                    self.stats["compressed_from_mem_to_l1"][layer_idx] += overflow
                    changed = True
            
            # 压缩层：使用学习的压缩器
            elif overflow >= layer_strides[0]:  # ✅ 使用stride1而非硬编码4
                stride1 = layer_strides[0]  # 使用该层的stride
                compress_num = overflow // stride1  # 压缩的个数（整除）
                compress_len = compress_num * stride1  # 实际压缩的长度
                
                # 先初始化为None，防止未定义
                k_comp = v_comp = None
                loss_dict = None
                
                if compress_len > 0:
                    # 选出需要压缩的量
                    old_comk = self.mem_k[layer_idx][:, :, :compress_len, :]
                    old_comv = self.mem_v[layer_idx][:, :, :compress_len, :]
                    
                    # 获取query（训练模式下需要）
                    query = None
                    if self.training_mode and new_k is not None:
                        query = new_k[:, :, -1:, :]  # [bs, heads, 1, head_dim]
                    
                    # 执行一级压缩
                    k_comp, v_comp, loss_dict = self._compress_chunk(
                        old_comk, old_comv, stride1, level=0, query=query, layer_idx=layer_idx
                    )
                    
                    # 累积损失（保持梯度链）
                    if loss_dict is not None and loss_dict.get('total') is not None:
                        if isinstance(self.accumulated_loss['total'], torch.Tensor):
                            self.accumulated_loss['total'] = self.accumulated_loss['total'] + loss_dict['total']
                        else:
                            # 第一次累积，从float变为tensor
                            self.accumulated_loss['total'] = loss_dict['total']
                        self.accumulated_loss['attn_dist'] += loss_dict.get('attn_dist', 0.0)
                        self.accumulated_loss['attn_output'] += loss_dict.get('attn_output', 0.0)
                        self.accumulated_loss['count'] += 1
                    
                if k_comp is not None and v_comp is not None:
                    # 将压缩后的数据追加到一级压缩层
                    self.level_k[0][layer_idx] = self._append(self.level_k[0][layer_idx], k_comp)
                    self.level_v[0][layer_idx] = self._append(self.level_v[0][layer_idx], v_comp)
                    self.stats["l1_compress_events"] += 1
                    self.stats["compressed_from_mem_to_l1"][layer_idx] += k_comp.shape[2]
                
                # 移除已压缩的部分（而不是截断到固定mem_len，避免丢失overflow中未压缩的token）
                self.mem_k[layer_idx] = self._take_suffix(self.mem_k[layer_idx], compress_len)
                self.mem_v[layer_idx] = self._take_suffix(self.mem_v[layer_idx], compress_len)
                changed = True  # mem 变化会影响 past            # 二级压缩：仅对压缩层处理，level_k[0] 溢出时压缩到 level_k[1]
            # 非压缩层的L1如果溢出，移动到L2但不压缩
            if self.num_levels > 1 and self.level_k[0][layer_idx] is not None:
                level1_overflow = self.level_k[0][layer_idx].shape[2] - self.level_caps[0]
                
                # 非压缩层：L1溢出时直接移动到L2（不压缩）
                if layer_idx not in self.compress_layers:
                    if level1_overflow > 0:
                        # 直接将溢出部分移到L2（完整保存）
                        overflow_k = self.level_k[0][layer_idx][:, :, :level1_overflow, :]
                        overflow_v = self.level_v[0][layer_idx][:, :, :level1_overflow, :]
                        
                        self.level_k[1][layer_idx] = self._append(self.level_k[1][layer_idx], overflow_k)
                        self.level_v[1][layer_idx] = self._append(self.level_v[1][layer_idx], overflow_v)
                        
                        # L1保留最新的部分
                        self.level_k[0][layer_idx] = self._take_suffix(self.level_k[0][layer_idx], level1_overflow)
                        self.level_v[0][layer_idx] = self._take_suffix(self.level_v[0][layer_idx], level1_overflow)
                        
                        self.stats["compressed_from_l1_to_l2"][layer_idx] += level1_overflow
                        changed = True
                
                # 压缩层：使用学习的压缩器
                elif level1_overflow >= (layer_strides[1] if len(layer_strides) > 1 else 1):  # ✅ 使用stride2而非硬编码4
                    stride2 = layer_strides[1] if len(layer_strides) > 1 else 1  # 使用该层的L2 stride
                    compress_num_l2 = level1_overflow // stride2  # 二级压缩的个数（整除）
                    compress_len_l2 = compress_num_l2 * stride2  # 实际压缩的长度
                    
                    # 先初始化为None，防止未定义
                    k_comp_l2 = v_comp_l2 = None
                    loss_dict_l2 = None
                    
                    if compress_len_l2 > 0:
                        if self.enable_debug and layer_idx == 0:
                            print(f"[L2 Compress] Layer {layer_idx}: L1 overflow={level1_overflow}, compressing {compress_len_l2} tokens")
                        
                        # 选出需要压缩的量（从 level_k[0] 的前部取出）
                        old_comk_l2 = self.level_k[0][layer_idx][:, :, :compress_len_l2, :]
                        old_comv_l2 = self.level_v[0][layer_idx][:, :, :compress_len_l2, :]
                        
                        # 获取query（训练模式下需要）
                        query_l2 = None
                        if self.training_mode and new_k is not None:
                            query_l2 = new_k[:, :, -1:, :]
                        
                        # 执行二级压缩
                        k_comp_l2, v_comp_l2, loss_dict_l2 = self._compress_chunk(
                            old_comk_l2, old_comv_l2, stride2, level=1, query=query_l2, layer_idx=layer_idx
                        )
                        
                        # 累积损失（保持梯度链 - infer函数）
                        if loss_dict_l2 is not None and loss_dict_l2.get('total') is not None:
                            if isinstance(self.accumulated_loss['total'], torch.Tensor):
                                self.accumulated_loss['total'] = self.accumulated_loss['total'] + loss_dict_l2['total']
                            else:
                                self.accumulated_loss['total'] = loss_dict_l2['total']
                            self.accumulated_loss['attn_dist'] += loss_dict_l2.get('attn_dist', 0.0)
                            self.accumulated_loss['attn_output'] += loss_dict_l2.get('attn_output', 0.0)
                            self.accumulated_loss['count'] += 1
                        
                        if k_comp_l2 is not None and v_comp_l2 is not None:
                            # 将压缩后的数据追加到二级压缩层
                            self.level_k[1][layer_idx] = self._append(self.level_k[1][layer_idx], k_comp_l2)
                            self.level_v[1][layer_idx] = self._append(self.level_v[1][layer_idx], v_comp_l2)
                            self.stats["l2_compress_events"] += 1
                            self.stats["compressed_from_l1_to_l2"][layer_idx] += k_comp_l2.shape[2]
                            
                            if self.enable_debug and layer_idx == 0:
                                l2_total = self.level_k[1][layer_idx].shape[2]
                                print(f"[L2 Compress] Layer {layer_idx}: Added {k_comp_l2.shape[2]} tokens to L2, L2 total={l2_total}")
                        
                        # 截断 level_k[0]，保留最后 level_caps[0] 个 token
                        self.level_k[0][layer_idx] = self._truncate_keep_tail(self.level_k[0][layer_idx], self.level_caps[0])
                        self.level_v[0][layer_idx] = self._truncate_keep_tail(self.level_v[0][layer_idx], self.level_caps[0])
                        changed = True  # level1 变化也会影响 past

            self.prev_total_len[layer_idx] = total_seq
            self._update_layer_lengths_stats(layer_idx)

        # 轻量 sanity 检查
        self.sanity_check()
        return changed

    def sanity_check(self):
        """长度越界与 NaN 检查（仅打印警告，不抛异常）"""
        # 在推理模式且非debug模式下跳过检查，避免大量日志
        if not self.training_mode and not self.enable_debug:
            return
            
        for i in range(self.n_layers):
            mem_len_now = 0 if self.mem_k[i] is None else self.mem_k[i].shape[2]
            layer_mem_len = self.mem_len_per_layer[i]
            if mem_len_now > layer_mem_len + 8:
                print(f"[WARN] mem_len exceeded at layer {i}: {mem_len_now} > {layer_mem_len}")

            if self.num_levels > 0 and self.level_k[0][i] is not None:
                l1_len_now = self.level_k[0][i].shape[2]
                if l1_len_now > self.level_caps[0] + 64:
                    print(f"[WARN] L1 cap exceeded at layer {i}: {l1_len_now} > {self.level_caps[0]}")

            if self.num_levels > 1 and self.level_k[1][i] is not None:
                l2_len_now = self.level_k[1][i].shape[2]
                if l2_len_now > 4 * max(1, self.level_caps[0]):
                    print(f"[WARN] L2 unusually large at layer {i}: {l2_len_now}")

            # NaN/Inf 检查（抽样）
            for buf in (self.mem_k[i], self.mem_v[i]):
                if buf is None: continue
                if not torch.isfinite(buf[..., -1:, :]).all():
                    print(f"[WARN] Non-finite values in mem at layer {i}")

    def get_stats(self) -> Dict[str, Any]:
        """返回一个可读的统计快照"""
        for i in range(self.n_layers):
            self._update_layer_lengths_stats(i)
        return dict(self.stats)

    def build_compressed_past(self):
        """
        构建喂给模型的"压缩后" past：
        时间顺序： L2(最老已压缩) → L1(已压缩) → mem(最新)
        """
        cache = DynamicCache()
        for i in range(self.n_layers):
            # 按时间顺序拼接：旧→新
            k = None
            v = None
            
            # L2 压缩段（最老）
            if self.num_levels > 1 and self.level_k[1][i] is not None:
                k = self.level_k[1][i]
                v = self.level_v[1][i]
            
            # L1 压缩段
            if self.level_k[0][i] is not None:
                k = self._cat(k, self.level_k[0][i])
                v = self._cat(v, self.level_v[0][i])
            
            # mem（最新）
            k = self._cat(k, self.mem_k[i])
            v = self._cat(v, self.mem_v[i])

            cache.key_cache.append(k)
            cache.value_cache.append(v)
        return cache

    def build_compressed_past_infer(self):
        """
        构建喂给模型的"压缩后" past：
        时间顺序： L2(最老已压缩) → L1(已压缩) → mem(最新)
        
        新策略：允许各层长度不同
        - 非压缩层（前24层）：保留完整历史（mem_len=100000，实际长度可能很长）
        - 压缩层（后4层）：压缩后的短序列（mem_len=512，经过16x压缩）
        """
        cache = DynamicCache()
        
        for i in range(self.n_layers):
            # 按时间顺序拼接：旧→新
            k = None
            v = None
            
            # L2 压缩段（最老）
            if self.num_levels > 1 and self.level_k[1][i] is not None:
                k = self.level_k[1][i]
                v = self.level_v[1][i]
            
            # L1 压缩段
            if self.level_k[0][i] is not None:
                k = self._cat(k, self.level_k[0][i])
                v = self._cat(v, self.level_v[0][i])
            
            # mem（最新）
            k = self._cat(k, self.mem_k[i])
            v = self._cat(v, self.mem_v[i])
            
            # 直接添加，不做min_len对齐
            # 前24层：可能有8K+ tokens（完整历史）
            # 后4层：可能只有512左右（压缩后）
            cache.key_cache.append(k)
            cache.value_cache.append(v)
        
        return cache

    def sync_prev_total_len_from_current_past(self, past_key_values):
        """
        根据当前构建好的 past 同步 prev_total_len，
        以便下次增量更新时能正确计算 delta。
        
        这个方法应该在 build_compressed_past_infer() 之后调用。
        """
        if past_key_values is None:
            return
        
        if hasattr(past_key_values, "key_cache"):
            k_list = past_key_values.key_cache
        else:
            k_list = [kv[0] for kv in past_key_values]
        
        for layer_idx in range(self.n_layers):
            self.prev_total_len[layer_idx] = k_list[layer_idx].shape[2]

    def update_from_model_past_infer(self, past_key_values, input_past_len=None) -> bool:
        """
        将模型返回的最新 past 增量喂入 CKVC，
        并根据 mem_len / caps / strides 进行批处理压缩。
        
        Args:
            past_key_values: 模型返回的完整past
            input_past_len: 我们传入模型的past长度（如果提供，用于精确计算delta）
        
        """
        changed = False
        if past_key_values is None:
            return changed

        if hasattr(past_key_values, "key_cache"):
            k_all_list = past_key_values.key_cache
            v_all_list = past_key_values.value_cache
        else:
            k_all_list = [kv[0] for kv in past_key_values]
            v_all_list = [kv[1] for kv in past_key_values]

        if self.device is None and k_all_list:
            self.device = k_all_list[0].device

        self.stats["events"] += 1

        for layer_idx in range(self.n_layers):
            k_all = k_all_list[layer_idx]
            v_all = v_all_list[layer_idx]
            total_seq = k_all.shape[2]
            # if(layer_idx == 1):
            #     print(f"total:{total_seq}")

            # 使用input_past_len精确计算delta（新增的token数）
            if input_past_len is not None:
                delta = total_seq - input_past_len
            else:
                delta = total_seq - self.prev_total_len[layer_idx]
            # print(f"delta:{delta}")
            if delta <= 0:
                continue
            
            new_k = k_all[:, :, -delta:, :].contiguous()
            new_v = v_all[:, :, -delta:, :].contiguous()
            self.stats["tokens_pushed_raw_per_layer"][layer_idx] += int(delta)
            
            # 先拼到 mem
            self.mem_k[layer_idx] = self._append(self.mem_k[layer_idx], new_k)
            self.mem_v[layer_idx] = self._append(self.mem_v[layer_idx], new_v)

            # 一级处理：使用各层独立的mem_len
            layer_mem_len = self.mem_len_per_layer[layer_idx]
            layer_strides = self.compress_strides_per_layer[layer_idx]
            overflow = self.mem_k[layer_idx].shape[2] - layer_mem_len
            
            # 非压缩层：将溢出部分完整移到L1（stride=1，实际不压缩）
            if layer_idx not in self.compress_layers:
                if overflow > 0:
                    # 将溢出的部分移到L1（不压缩，完整保存）
                    overflow_k = self.mem_k[layer_idx][:, :, :overflow, :]
                    overflow_v = self.mem_v[layer_idx][:, :, :overflow, :]
                    
                    # 添加到L1
                    self.level_k[0][layer_idx] = self._append(self.level_k[0][layer_idx], overflow_k)
                    self.level_v[0][layer_idx] = self._append(self.level_v[0][layer_idx], overflow_v)
                    
                    # mem保留最新的部分
                    self.mem_k[layer_idx] = self._take_suffix(self.mem_k[layer_idx], overflow)
                    self.mem_v[layer_idx] = self._take_suffix(self.mem_v[layer_idx], overflow)
                    
                    self.stats["compressed_from_mem_to_l1"][layer_idx] += overflow
                    changed = True
            
            # 压缩层：使用学习的压缩器
            elif overflow >= layer_strides[0]:  # 使用stride1而非硬编码4
                stride1 = layer_strides[0]  # 使用该层的stride
                compress_num = overflow // stride1
                compress_len = compress_num * stride1
                
                # 先初始化为None，防止未定义
                old_k1 = old_v1 = None
                loss_dict = None
                
                if compress_len > 0:
                    old_k = self.mem_k[layer_idx][:, :, :compress_len, :]
                    old_v = self.mem_v[layer_idx][:, :, :compress_len, :]
                    
                    # 获取query（训练模式下需要）
                    query = None
                    if self.training_mode and new_k is not None:
                        # 使用最新的token作为query
                        query = new_k[:, :, -1:, :]  # [bs, heads, 1, head_dim]
                    
                    old_k1, old_v1, loss_dict = self._compress_chunk(
                        old_k, old_v, stride1, level=0, query=query, layer_idx=layer_idx
                    )
                    
                    # 累积损失（安全处理tensor和float两种情况）
                    if loss_dict is not None and loss_dict.get('total') is not None:
                        if isinstance(self.accumulated_loss['total'], torch.Tensor):
                            self.accumulated_loss['total'] = self.accumulated_loss['total'] + loss_dict['total']
                        else:
                            self.accumulated_loss['total'] = loss_dict['total']
                        self.accumulated_loss['attn_dist'] += loss_dict.get('attn_dist', 0.0)
                        self.accumulated_loss['attn_output'] += loss_dict.get('attn_output', 0.0)
                        self.accumulated_loss['count'] += 1

                if old_k1 is not None and old_v1 is not None:
                    self.level_k[0][layer_idx] = self._append(self.level_k[0][layer_idx], old_k1)
                    self.level_v[0][layer_idx] = self._append(self.level_v[0][layer_idx], old_v1)
                    self.stats["l1_compress_events"] += 1
                    self.stats["compressed_from_mem_to_l1"][layer_idx] += old_k1.shape[2]

                # 移除已压缩的部分（而不是截断到固定mem_len，避免丢失overflow中未压缩的token）
                self.mem_k[layer_idx] = self._take_suffix(self.mem_k[layer_idx], compress_len)
                self.mem_v[layer_idx] = self._take_suffix(self.mem_v[layer_idx], compress_len)
                changed = True  # mem 变化会影响 past            # 二级处理：区分压缩层和非压缩层（与init路径保持一致）
            if self.num_levels > 1 and self.level_k[0][layer_idx] is not None:
                level1_overflow = self.level_k[0][layer_idx].shape[2] - self.level_caps[0]
                
                # 非压缩层：L1溢出时直接移动到L2（不压缩）
                if layer_idx not in self.compress_layers:
                    if level1_overflow > 0:
                        # 直接将溢出部分移到L2（完整保存）
                        overflow_k = self.level_k[0][layer_idx][:, :, :level1_overflow, :]
                        overflow_v = self.level_v[0][layer_idx][:, :, :level1_overflow, :]
                        
                        self.level_k[1][layer_idx] = self._append(self.level_k[1][layer_idx], overflow_k)
                        self.level_v[1][layer_idx] = self._append(self.level_v[1][layer_idx], overflow_v)
                        
                        # L1保留最新的部分
                        self.level_k[0][layer_idx] = self._truncate_keep_tail(self.level_k[0][layer_idx], self.level_caps[0])
                        self.level_v[0][layer_idx] = self._truncate_keep_tail(self.level_v[0][layer_idx], self.level_caps[0])
                        
                        self.stats["compressed_from_l1_to_l2"][layer_idx] += level1_overflow
                        changed = True
                
                # 压缩层：使用学习的压缩器
                elif level1_overflow >= self.compress_strides[1]:  # 使用stride2而非硬编码4
                    stride2 = self.compress_strides[1]
                    compress_num_l2 = level1_overflow // stride2
                    compress_len_l2 = compress_num_l2 * stride2
                    
                    # 先初始化为None，防止未定义
                    k_comp_l2 = v_comp_l2 = None
                    loss_dict_l2 = None
                    
                    if compress_len_l2 > 0:
                        if self.enable_debug and layer_idx == 0:
                            print(f"[L2 Compress Infer] Layer {layer_idx}: L1 overflow={level1_overflow}, compressing {compress_len_l2} tokens")
                        
                        # 选出需要压缩的量（从 level_k[0] 的前部取出）
                        old_comk_l2 = self.level_k[0][layer_idx][:, :, :compress_len_l2, :]
                        old_comv_l2 = self.level_v[0][layer_idx][:, :, :compress_len_l2, :]
                        
                        # 获取query（训练模式下需要）
                        query_l2 = None
                        if self.training_mode and new_k is not None:
                            query_l2 = new_k[:, :, -1:, :]
                        
                        # 执行二级压缩
                        k_comp_l2, v_comp_l2, loss_dict_l2 = self._compress_chunk(
                            old_comk_l2, old_comv_l2, stride2, level=1, query=query_l2, layer_idx=layer_idx
                        )
                        
                        # 累积损失（保持梯度链）
                        if loss_dict_l2 is not None and loss_dict_l2.get('total') is not None:
                            if isinstance(self.accumulated_loss['total'], torch.Tensor):
                                self.accumulated_loss['total'] = self.accumulated_loss['total'] + loss_dict_l2['total']
                            else:
                                self.accumulated_loss['total'] = loss_dict_l2['total']
                            self.accumulated_loss['attn_dist'] += loss_dict_l2.get('attn_dist', 0.0)
                            self.accumulated_loss['attn_output'] += loss_dict_l2.get('attn_output', 0.0)
                            self.accumulated_loss['count'] += 1
                        
                        if k_comp_l2 is not None and v_comp_l2 is not None:
                            # 将压缩后的数据追加到二级压缩层
                            self.level_k[1][layer_idx] = self._append(self.level_k[1][layer_idx], k_comp_l2)
                            self.level_v[1][layer_idx] = self._append(self.level_v[1][layer_idx], v_comp_l2)
                            self.stats["l2_compress_events"] += 1
                            self.stats["compressed_from_l1_to_l2"][layer_idx] += k_comp_l2.shape[2]
                            
                            if self.enable_debug and layer_idx == 0:
                                l2_total = self.level_k[1][layer_idx].shape[2]
                                print(f"[L2 Compress Infer] Layer {layer_idx}: Added {k_comp_l2.shape[2]} tokens to L2, L2 total={l2_total}")
                        
                        # 截断 level_k[0]，保留最后 level_caps[0] 个 token
                        self.level_k[0][layer_idx] = self._truncate_keep_tail(self.level_k[0][layer_idx], self.level_caps[0])
                        self.level_v[0][layer_idx] = self._truncate_keep_tail(self.level_v[0][layer_idx], self.level_caps[0])
                        changed = True  # level1 变化也会影响 past


            self.prev_total_len[layer_idx] = total_seq
            self._update_layer_lengths_stats(layer_idx)

        # 轻量 sanity 检查
        self.sanity_check()
        return changed
    
    def enable_training_mode(self):
        """启用训练模式（计算注意力损失）"""
        self.training_mode = True
        device = self.device if self.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accumulated_loss = {
            'total': torch.tensor(0.0, device=device, requires_grad=True),
            'attn_dist': 0.0,
            'attn_output': 0.0,
            'count': 0
        }
    
    def disable_training_mode(self):
        """禁用训练模式"""
        self.training_mode = False
    
    def get_accumulated_loss(self):
        """获取累积的损失"""
        if self.accumulated_loss['count'] == 0:
            return None
        
        # 如果没有累积任何有效损失，返回None
        if not isinstance(self.accumulated_loss['total'], torch.Tensor):
            return None
        
        avg_loss = {
            'total': self.accumulated_loss['total'] / self.accumulated_loss['count'],
            'attn_dist': self.accumulated_loss['attn_dist'] / self.accumulated_loss['count'],
            'attn_output': self.accumulated_loss['attn_output'] / self.accumulated_loss['count'],
            'count': self.accumulated_loss['count']
        }
        return avg_loss
    
    def reset_accumulated_loss(self):
        """重置累积损失"""
        device = self.device if self.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.accumulated_loss = {
            'total': torch.tensor(0.0, device=device, requires_grad=True),
            'attn_dist': 0.0,
            'attn_output': 0.0,
            'count': 0
        }
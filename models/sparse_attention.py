"""Block-sparse self-attention for VOSR2: ``attn_type`` full | sparse | local.

``SparseProcessAttnAigc`` (sparse: per 8x8-token window, keep the 8 key windows with the
highest mean-q . mean-k score), ``create_window_mask`` and ``SparseProcessAttnAigc_Local0408``
(local: 3x3 neighbourhood of windows) are copied verbatim from the user's on-device
implementation (their models/bmm.py) and must not be edited. ``reshape1D``/``unreshape1D``
are the bodies of the methods of the same name in their transformer, without ``self``.

Not part of the pasted code, written here: the ``ScaledDotProductAttnAigc`` base class
(only ``heads``/``dim_head`` are used by the copied forwards), the NPU imports, and
``windowed_attention``, which puts VOSR2's row-major tokens into window order around the call.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import is_torch_npu_available

if is_torch_npu_available():
    import torch_npu
else:
    torch_npu = None

ATTN_TYPES = ('full', 'sparse', 'local')
WINDOW = 8  # the copied classes hard-code block_lenth = 64 tokens = one 8x8 window


class ScaledDotProductAttnAigc(nn.Module):
    """Stand-in for the user's base class, which was not shared."""
    def __init__(self, dim_head=None, heads=None):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                                              is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)


# ---- Copied verbatim from the user's transformer (methods, ``self`` dropped) ----

def reshape1D(hidden_states, new_H, new_W, block_lenth_2D, new_img_len):
    bs = hidden_states.shape[0]
    return hidden_states.view(bs*new_H, block_lenth_2D, new_W, -1).transpose(1, 2).reshape(bs, new_img_len, -1)

def unreshape1D(hidden_states, new_H, new_W, block_lenth_2D, new_img_len):
    bs = hidden_states.shape[0]
    return hidden_states.reshape(bs*new_H, new_W, block_lenth_2D, -1).transpose(1, 2).reshape(bs, new_img_len, -1)


# ---- Copied verbatim from the user's models/bmm.py ----

class SparseProcessAttnAigc(ScaledDotProductAttnAigc):

    def __init__(self, dim_head=None, heads=None):
        super().__init__(dim_head, heads)

    def split_and_squeeze(self, ori_tensor, block_lenth):
        B, n, L, C = ori_tensor.shape
        new_L = L // block_lenth
        assert L % block_lenth == 0, f'B, N, L, C: {B}, {n}, {L}, {C}'
        mean_tensor = ori_tensor.view(B, n, new_L, block_lenth, C).mean(dim=3)

        return mean_tensor

    def scale_dot(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

        return super().forward(
            query, key, value, attn_mask, \
            dropout_p, is_causal, scale, enable_gqa
        )


    def forward(self, query, key, value, img_len, batch_size, sparse_ratio): #### qkv after retory
        # add by yulei.
        ''' args '''
        block_lenth = 64  # num_block = 4096 // 32 = 128 (1024 * 1024)  256 // 32 = 8 (256, 256)
        H, W = img_len
        new_img_len = H * W

        query_image = query[:, :, -new_img_len:, :]
        key_image = key[:, :, -new_img_len:, :]
        N_text = query.shape[2] - query_image.shape[2]

        num_block = (query_image.shape[2]) // block_lenth
        topK = int(num_block * sparse_ratio)  # k=0.75 means mask ratio is 75%, equal to kv_compress_ratio=2
        topK = num_block - 8 # 通路上最大计算只能取8，因此反算topK有此限制

        query_image_block = self.split_and_squeeze(query_image, block_lenth=block_lenth)
        key_image_block = self.split_and_squeeze(key_image, block_lenth=block_lenth)

        similarity_matrix = torch.matmul(query_image_block, key_image_block.transpose(-1, -2))
        # print(similarity_matrix)

        # print(f"similarity_matrix: {similarity_matrix.shape}")

        _, top_index = (-similarity_matrix).topk(k=topK, dim=-1)
        # print(f"top_index: {top_index.shape}")
        mask = torch.zeros(batch_size, query.shape[1], num_block, num_block).to(query_image.device)
        # print(f"mask: {mask.shape}")
        mask = mask.scatter_(3, top_index, float('-inf'))
        # mask = mask.repeat_interleave(block_lenth, dim=2).repeat_interleave(block_lenth, dim=3)
        mask = mask.repeat_interleave(block_lenth, dim=2)
        mask_shape = mask.shape
        mask = mask.unsqueeze(4).expand(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3], block_lenth)
        mask = mask.reshape(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3] * block_lenth)

        # if encoder_hidden_states is not None:
        mask = torch.nn.functional.pad(mask, (N_text, 0, N_text, 0))
        mask = mask.to(torch.bool)
        # print(f"final mask: {mask.shape}")

        # add by dkp.
        ori_hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.logical_not(), dropout_p=0.0,
                                                           is_causal=False)  # add mask input,
        return ori_hidden_states

# 2026.04.08 新增的patch mask
def create_window_mask(width, height, window_side_len):
    num_tokens = width*height      # 64x48=3072
    offset = (window_side_len-1) // 2   # 假设window_side_len=3, offset=1

    mask = torch.zeros(num_tokens, num_tokens)   # (3072, 3072)

    for i in range(mask.shape[0]):
        h = i // width  # [0,0,...,1,,..,63,63]  当前 token 的行坐标
        w = i % width   # [0,1,..,63,0,1,..63...]  当前 token 的列坐标

        window = torch.ones(window_side_len, window_side_len)   # (3, 3)
        query_mask = torch.zeros(height, width)   # (48, 64)

        query_mask[
            max(0, h-offset):min(height, h+offset+1),
            max(0, w-offset):min(width, w+offset+1)
        ] = window[
            0:min(height, h+offset+1)-max(0, h-offset),
            0:min(width, w+offset+1)-max(0, w-offset)
        ]

        mask[i] = query_mask.flatten()

    return mask


class SparseProcessAttnAigc_Local0408(ScaledDotProductAttnAigc):

    def __init__(self, dim_head=None, heads=None, patch_size=3):
        super().__init__(dim_head, heads)
        self.patch_size = patch_size

    def split_and_squeeze(self, ori_tensor, block_lenth):
        B, n, L, C = ori_tensor.shape
        new_L = L // block_lenth
        assert L % block_lenth == 0, f'B, N, L, C: {B}, {n}, {L}, {C}'
        mean_tensor = ori_tensor.view(B, n, new_L, block_lenth, C).mean(dim=3)

        return mean_tensor

    def scale_dot(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

        return super().forward(
            query, key, value, attn_mask, \
            dropout_p, is_causal, scale, enable_gqa
        )

    def forward(self, query, key, value, img_len, batch_size, sparse_ratio): #### qkv after retory
        # add by yulei.
        ''' args '''
        block_lenth = 64  # num_block = 4096 // 32 = 128 (1024 * 1024)  256 // 32 = 8 (256, 256)
        block_lenth_2D = int(math.sqrt(block_lenth))
        H, W = img_len
        new_img_len = H * W

        ''' 1.获取图像部分的 QK '''

        query_image = query[:, :, -new_img_len:, :]
        key_image = key[:, :, -new_img_len:, :]
        N_text = query.shape[2] - query_image.shape[2]

        W_block = W // block_lenth_2D
        H_block = H // block_lenth_2D
        num_block = W_block * H_block  # 此时 num_block 将动态等于 48 (或其他尺度)


        # 2026.04.08 换mask
        # window_mask_2d = create_window_mask(8, 8, 5).to(query_image.device)
        window_mask_2d = create_window_mask(W_block, H_block, self.patch_size).to(query_image.device)

        # 创建匹配原来维度的 4D mask：[B, Heads, num_block, num_block]
        mask = torch.zeros(batch_size, query.shape[1], num_block, num_block, device=query_image.device)

        # 将 2D mask 广播到 4D，并把 window_mask_2d 中为 0 (不需要计算) 的位置赋值为 -inf
        # 这样就完美兼容了你原代码“0表示计算，-inf表示Mask”的逻辑
        mask_condition = window_mask_2d.view(1, 1, num_block, num_block).expand_as(mask)
        mask[mask_condition == 0] = float('-inf')
        mask[mask_condition == 1] = 0

        # mask = mask.scatter_(3, top_index, float('-inf'))  # 把最小的那些index mask为 -inf

        mask = mask.repeat_interleave(block_lenth, dim=2)
        mask_shape = mask.shape
        mask = mask.unsqueeze(4).expand(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3], block_lenth)
        mask = mask.reshape(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3] * block_lenth)

        ''' 5.还得把mask扩展一下 以匹配text token '''
        # if encoder_hidden_states is not None:
        mask = torch.nn.functional.pad(mask, (N_text, 0, N_text, 0))
        mask = mask.to(torch.bool)
        # print(f"final mask: {mask.shape}")

        # add by dkp.
        if is_torch_npu_available() and query.dtype in (torch.float16, torch.bfloat16):
            ori_hidden_states = torch_npu.npu_fusion_attention(
                query,
                key,
                value,
                self.heads,
                atten_mask=mask,  # add
                input_layout="BNSD",
                pse=None,
                scale=1.0 / math.sqrt(query.shape[-1]),
                pre_tockens=65536,
                next_tockens=65536,
                keep_prob=1.0,
                sync=False,
                inner_precise=0,
            )[0]
        else:
            # 警告：NPU上mask为1的区域不计算attention，而GPU上mask为0的区域不计算attention，所以这里需要对mask矩阵取反
            ori_hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.logical_not(), dropout_p=0.0,
                                                           is_causal=False)  # add mask input,
        return ori_hidden_states


# ---- VOSR2 glue ----

def build_sparse_attn(attn_type, dim_head, heads):
    """The user's module for ``attn_type``; None for full attention."""
    if attn_type not in ATTN_TYPES:
        raise ValueError(f'attn_type must be one of {ATTN_TYPES}, got {attn_type!r}')
    if attn_type == 'full':
        return None
    cls = SparseProcessAttnAigc if attn_type == 'sparse' else SparseProcessAttnAigc_Local0408
    return cls(dim_head=dim_head, heads=heads)


def windowed_attention(sparse_attn, q, k, v):
    """q, k, v: (B, heads, N, head_dim) after RoPE, N a row-major square token grid.

    Reorders the tokens so every 64 consecutive tokens form one 8x8 window (the user's
    reshape1D), calls their module, and restores the row-major order.
    """
    b, heads, n, d = q.shape
    side = math.isqrt(n)
    if side * side != n or side % WINDOW:
        raise ValueError(f'Block-sparse attention needs a square token grid with sides divisible by {WINDOW}; '
                         f'got {n} tokens')
    windows = side // WINDOW
    if isinstance(sparse_attn, SparseProcessAttnAigc) and windows * windows <= 8:
        # Keeping 8 windows out of at most 8 is full attention; the copied top-k
        # (topk(num_block - 8)) cannot run below 8 windows, e.g. on hourglass merged tokens.
        return F.scaled_dot_product_attention(q, k, v)

    def to_windows(t):
        # reshape1D uses .view, so it needs contiguous tokens (v comes from a permuted qkv split).
        return reshape1D(t.contiguous().view(b * heads, n, d), windows, windows, WINDOW, n).view(b, heads, n, d)
    # sparse_ratio is unused by both copied classes (top-k overwrites it); 0.8333 is the user's default.
    out = sparse_attn(to_windows(q), to_windows(k), to_windows(v), (side, side), b, 0.8333)
    return unreshape1D(out.contiguous().view(b * heads, n, d), windows, windows, WINDOW, n).view(b, heads, n, d)

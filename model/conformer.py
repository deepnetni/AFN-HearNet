from typing import Optional, Union
import torch
from torch import Tensor, nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange


# source: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, context=None, mask=None, context_mask=None):
        """

        :param x: B,N,D
        :param context: B,N_,D, N_ should equal N in x
        :param mask: N,N_; 1 is effective.
        :param context_mask:
        :returns:

        """
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale  # BHNN_

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)  # n is the T
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")  # NN_
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)  # NN_D
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, x.new_ones(*x.shape[:2]))  # B,N
            if not has_context:
                context_mask = default(context_mask, mask)
            else:  # has context
                context_mask = default(context_mask, x.new_ones(*context.shape[:2]))

            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(context_mask, "b j -> b () () j")
            # masked_fill replace the value where cond is True
            dots.masked_fill_(~mask.bool(), mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        conv_causal_mode=True,
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.0,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn = Attention(
        #     dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        # )
        self.attn_pre_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal_mode,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        # self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def get_mask(self, x, attn_wlen: Optional[int] = None):
        """should be call outside
        mask the componet where the mask is True.

        x with shape (B,T,H)

        return: B,T,T
        """
        nT = x.shape[-2]

        mask = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
        if attn_wlen is not None:
            # assert attn_wlen >= 1
            mask_prev = torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-attn_wlen)
            mask = mask + mask_prev
        return mask.bool()

    def forward(
        self,
        x,
        mask: Optional[torch.Tensor] = None,
        need_weights=True,
    ):
        x = self.ff1(x) + x
        # x = self.attn(x, mask=mask) + x

        x = self.attn_pre_norm(x)
        x_mhsa, attn = self.attn(
            x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=mask
        )
        x = x + x_mhsa

        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x, attn


class DiTConformerBlock(nn.Module):
    """cross-attention conditional conformer block"""

    def __init__(
        self,
        *,
        dim,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        conv_causal_mode=True,
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.0,
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn_pre_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, batch_first=True, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=conv_causal_mode,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        # self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift

    def get_mask(self, x, attn_wlen: Optional[int] = None):
        """should be call outside
        mask the componet where the mask is True.

        x with shape (B,T,H)

        return: B,T,T
        """
        nT = x.shape[-2]

        mask = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
        if attn_wlen is not None:
            # assert attn_wlen >= 1
            mask_prev = torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-attn_wlen)
            mask = mask + mask_prev
        return mask.bool()

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights=True,
    ):
        """

        :param x: B,T,D
        :param cond: B,T,D
        :param mask: B,T,T
        :param need_weights:
        :returns: B,T,D; B,T,T

        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)

        x = self.modulate(x, shift_msa, scale_msa)
        x = self.ff1(x) + x
        x = self.attn_pre_norm(x)
        x_mhsa, attn = self.attn(
            x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=mask
        )
        x = x + x_mhsa * gate_msa

        x = self.conv(x) + x

        x = self.modulate(x, shift_mlp, scale_mlp)
        x = self.ff2(x) * gate_mlp + x
        x = self.post_norm(x)
        return x, attn


# class CrossAttention(nn.Module):
#     """Cross Attention block"""

#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head**-0.5
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim)

#         self.max_pos_emb = max_pos_emb
#         self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, context=None, mask=None, context_mask=None):
#         """

#         :param x: B,T,N
#         :param y: B,T,N
#         :param context:
#         :param mask:
#         :param context_mask:
#         :returns:

#         """
#         n, device, h, max_pos_emb, has_context = (
#             x.shape[-2],
#             x.device,
#             self.heads,
#             self.max_pos_emb,
#             exists(context),
#         )
#         context = context or x

#         q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

#         dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

#         # shaw's relative positional embedding
#         seq = torch.arange(n, device=device)
#         dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
#         dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
#         rel_pos_emb = self.rel_pos_emb(dist).to(q)
#         pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
#         dots = dots + pos_attn

#         if exists(mask) or exists(context_mask):
#             mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
#             context_mask = (
#                 default(context_mask, mask)
#                 if not has_context
#                 else default(context_mask, lambda: torch.ones(*context.shape[:2], device=device))
#             )
#             mask_value = -torch.finfo(dots.dtype).max
#             mask = rearrange(mask, "b i -> b () i ()") * rearrange(context_mask, "b j -> b () () j")
#             dots.masked_fill_(~mask, mask_value)

#         attn = dots.softmax(dim=-1)

#         out = einsum("b h i j, b h j d -> b h i d", attn, v)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.to_out(out)
#         return self.dropout(out)


# class CAttnCondConformerBlock(nn.Module):
#     """cross-attention conditional conformer block"""

#     def __init__(
#         self,
#         *,
#         dim,
#         heads=8,
#         ff_mult=4,
#         conv_expansion_factor=2,
#         conv_kernel_size=31,
#         conv_causal_mode=True,
#         attn_dropout=0.2,
#         ff_dropout=0.2,
#         conv_dropout=0.0,
#     ):
#         super().__init__()
#         self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
#         # self.attn = Attention(
#         #     dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
#         # )
#         self.attn_pre_norm = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim, num_heads=heads, batch_first=True, dropout=attn_dropout
#         )
#         self.conv = ConformerConvModule(
#             dim=dim,
#             causal=conv_causal_mode,
#             expansion_factor=conv_expansion_factor,
#             kernel_size=conv_kernel_size,
#             dropout=conv_dropout,
#         )
#         self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

#         # self.attn = PreNorm(dim, self.attn)
#         self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
#         self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

#         self.post_norm = nn.LayerNorm(dim)

#     def get_mask(self, x, attn_wlen: Optional[int] = None):
#         """should be call outside
#         mask the componet where the mask is True.

#         x with shape (B,T,H)

#         return: B,T,T
#         """
#         nT = x.shape[-2]

#         mask = torch.ones(nT, nT, device=x.device, dtype=torch.bool).triu_(1)  # TxT
#         if attn_wlen is not None:
#             # assert attn_wlen >= 1
#             mask_prev = torch.ones(nT, nT, device=x.device, dtype=torch.bool).tril_(-attn_wlen)
#             mask = mask + mask_prev
#         return mask.bool()

#     def forward(
#         self,
#         x: Tensor,
#         cond: Tensor,
#         mask: Optional[torch.Tensor] = None,
#         need_weights=True,
#     ):
#         x = self.ff1(x) + x

#         x = self.attn_pre_norm(x)
#         x_mhsa, attn = self.attn(
#             x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=mask
#         )
#         x = x + x_mhsa

#         x = self.conv(x) + x
#         x = self.ff2(x) + x
#         x = self.post_norm(x)
#         return x, attn


if __name__ == "__main__":
    inp = torch.randn(2, 10, 20)
    cond = torch.randn(2, 1, 20)
    net = ConformerBlock(dim=20, heads=4)
    # net = DiTConformerBlock(dim=20, heads=4)
    out, attn = net(inp, cond)
    print(out.shape, attn.shape)

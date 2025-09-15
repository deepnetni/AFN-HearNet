import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from model.conformer import ConformerBlock


class AMFTConformer(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.0,
    ):
        super().__init__()

        self.time_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=True,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.freq_conformer = ConformerBlock(
            dim=dim,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            conv_causal_mode=False,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
        )
        self.adaLN_modulation_f = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm1_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_f = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_f = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * ff_mult, dim),
        )

    @staticmethod
    def modulate(x, shift, scale):
        """

        :param x: b,t,c
        :param shift: b,c
        :param scale: b,c
        :returns:

        """
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def modulate_f(x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c, causal=False, wlen=None):
        """
        x: b,c,t,f
        c: b,f,c
        """

        nB = x.size(0)
        nT = x.size(-2)
        ##################
        # Freq Conformer #
        ##################

        # conditions, b,f,c
        (
            shift_msa_f,
            scale_msa_f,
            gate_msa_f,
            shift_mlp_f,
            scale_mlp_f,
            gate_mlp_f,
        ) = self.adaLN_modulation_f(c).chunk(6, dim=-1)

        shift_msa_f = shift_msa_f.repeat_interleave(nT, dim=0)  # bt,f,c
        scale_msa_f = scale_msa_f.repeat_interleave(nT, dim=0)
        gate_msa_f = gate_msa_f.repeat_interleave(nT, dim=0)
        shift_mlp_f = shift_mlp_f.repeat_interleave(nT, dim=0)
        scale_mlp_f = scale_mlp_f.repeat_interleave(nT, dim=0)
        gate_mlp_f = gate_mlp_f.repeat_interleave(nT, dim=0)

        x_f = einops.rearrange(x, "b c t f->(b t) f c")

        # bt,f,c * bt,f,c + bt,f,c
        x_ = self.modulate_f(self.norm1_f(x_f), shift_msa_f, scale_msa_f)
        x_, attn_f = self.freq_conformer(x_)
        x_f = x_ * gate_msa_f + x_f
        x_ = gate_mlp_f * self.mlp_f(
            self.modulate_f(self.norm2_f(x_f), shift_mlp_f, scale_mlp_f)
        )
        x_f = x_f + x_

        ####################
        # # Time Conformer #
        ####################
        if causal:
            mask = self.time_conformer.get_mask(x, wlen)
        else:
            mask = None

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c.view(-1, c.size(-1)).contiguous()).chunk(6, dim=1)

        x_t = einops.rearrange(x_f, "(b t) f c->(b f) t c", b=nB)

        # bf,c->bf,1,c x bf,t,c
        x_ = self.modulate(self.norm1(x_t), shift_msa, scale_msa)
        x_, attn_t = self.time_conformer(x_, mask=mask)
        x_t = x_ * gate_msa.unsqueeze(1) + x_t
        x_ = gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x_t), shift_mlp, scale_mlp)
        )
        x_t = x_t + x_
        x_ = einops.rearrange(x_t, "(b f) t c->b c t f", b=nB)

        return x_, (attn_f, attn_t)

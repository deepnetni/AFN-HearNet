import sys
from typing import List, Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Size, Tensor
from model.conv_stft import STFT
from model.AMFTConformer import AMFTConformer


class HLModule(nn.Module):
    def __init__(
        self,
        nbin=257,
        fs=16000,
        HL_freq=[250, 500, 1000, 2000, 4000, 8000],
        # fmt: off
        HL_freq_extend=torch.tensor([250, 375, 500, 625, 750, 1000, 1125, 1375,
                        1750, 2125, 2625, 3125, 3875, 4625, 5500, 6625]),
        freq_bands_range=[0, 250, 375, 500, 625, 750, 1000, 1250, 1625,
                          2000, 2375, 2875, 3500, 4250, 5125, 6125, 8001]
        # fmt: on
    ) -> None:
        super().__init__()

        self.freqs = torch.linspace(0, fs // 2, nbin)  # nbin, (fs//2) / (nbin)
        self.reso = fs // 2 / (nbin - 1)

        # sub-bands
        bands_filter = self._rectangular_filters(self.freqs, freq_bands_range)
        # hl_freq_ext = self.freqs if full else torch.tensor(HL_freq_extend)

        # HL curve index
        HL_curve, delta_x = self._HL_curve_idx(
            HL_freq=[0, *HL_freq], HL_freq_extend=HL_freq_extend
        )
        self.register_buffer("hl_freq", torch.tensor([0, *HL_freq]).float())

        self.register_buffer("bands_filter", bands_filter)
        self.register_buffer("HL_curve", HL_curve)
        self.register_buffer("delta_x", delta_x)

    @staticmethod
    def _HL_curve_idx(HL_freq, HL_freq_extend):
        """extend the Hearing Loss threshold to other freqency point.

        :param HL_freq: 7,
        :param HL_freq_extend: N,
        :returns: N,; N,

        """
        hl_freq = torch.tensor(HL_freq).float()

        curve_idx = torch.tensor([(x >= hl_freq).sum() - 1 for x in HL_freq_extend])
        delta_x = torch.tensor(
            [x - hl_freq[idx] for x, idx in zip(HL_freq_extend, curve_idx)]
        )
        return curve_idx, delta_x

    def _rectangular_filters(self, all_freqs, bands_range):
        nbands = len(bands_range) - 1
        bands_filter = torch.zeros((nbands, len(all_freqs)))
        bands_idx = [
            (i, ((all_freqs >= low) & (all_freqs < high)).nonzero(as_tuple=True)[0])
            for i, (low, high) in enumerate(zip(bands_range[:-1], bands_range[1:]))
        ]
        for idx in bands_idx:
            bands_filter[idx] = 1

        return bands_filter.permute(1, 0)  # nbin, nbands

    def _HL_LinearFitting(self, HL):
        diff_x = torch.diff(self.hl_freq)
        diff_y = torch.diff(HL)
        # print(HL.shape, diff_x.shape, "@", diff_y.shape)

        k = diff_y / diff_x  # B,nbands
        k = torch.concat([k, k.new_zeros(k.size(0), 1)], dim=-1)
        b = HL

        return k, b

    def extend_with_value(self, hl, T=None):
        """extend with self value

        :param hl: B,8
        :returns: B,1,T,nbin

        """
        m = int(250 / self.reso)
        bandarray = torch.tensor([0] + [(2**i) * m for i in range(hl.shape[1])]).to(
            hl.device
        )
        T = T or 1

        repeat_n = bandarray[1:] - bandarray[:-1]
        repeat_n[0] += 1

        expand_ht = (
            hl.repeat_interleave(repeat_n, dim=-1).unsqueeze(1).unsqueeze(1)
        )  # B,1,1,nbin
        expand_ht = expand_ht.repeat(1, 1, T, 1) / 100.0

        return expand_ht

    def extend_with_linear(self, hl, T=None) -> Tensor:
        """extend with self value

        :param hl: B,6
        :returns: B,1,T,nbin

        """
        T = T or 1
        hl = torch.concat([hl[:, (0,)], hl], dim=-1)

        k, b = self._HL_LinearFitting(hl)  # b,nbands(16)
        hl_ext = k[:, self.HL_curve] * self.delta_x + b[:, self.HL_curve]
        # print(k.shape, b.shape, hl_ext.shape, self.HL_curve.shape)

        expand_ht = hl_ext.unsqueeze(1).unsqueeze(1).repeat(1, 1, T, 1)  # b,1,nbands

        return expand_ht / 100


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)  # b,nc/r,h,w,r
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size]) -> None:
        super().__init__()

        self.norm = nn.Sequential(
            Rearrange("b c t f-> b t f c"),
            nn.LayerNorm(normalized_shape=normalized_shape),
            Rearrange("b t f c-> b c t f"),
        )

    def forward(self, x: Tensor):
        """
        x: b,c,t,f
        """
        return self.norm(x)


class DilatedDenseNet(nn.Module):
    def __init__(
        self,
        depth=4,
        in_channels=64,
        kernel_size=(2, 3),
    ):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        twidth = kernel_size[0]
        for i in range(self.depth):
            dil = 2**i
            pad_length = twidth + (dil - 1) * (twidth - 1) - 1

            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Sequential(
                    nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),  # lrtb
                    nn.Conv2d(
                        in_channels * (i + 1),
                        in_channels,
                        kernel_size=kernel_size,
                        dilation=(dil, 1),
                    ),
                    # nn.BatchNorm2d(in_channels),
                    LayerNorm(in_channels),
                    nn.PReLU(in_channels),
                ),
            )

        self.post = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1)))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "conv{}".format(i + 1))(skip)
            skip = torch.cat([out, skip], dim=1)

        return out


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 3), (1, 2), padding=(0, 1)),
            # nn.InstanceNorm2d(channels, affine=True),
            LayerNorm(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            # nn.InstanceNorm2d(channels, affine=True),
            LayerNorm(channels),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super().__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            # nn.InstanceNorm2d(num_channel),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, out_channel, (1, 2)),
            # nn.InstanceNorm2d(out_channel, affine=True),
            nn.BatchNorm2d(out_channel),
            nn.PReLU(out_channel),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (1, 1)),
            Rearrange("b c t f->b f t c"),
            nn.PReLU(num_features),
            Rearrange("b f t c->b c t f"),
        )

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.final_conv(x)
        return x


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.Conv2d(num_channel, num_channel, (1, 2)),
            # nn.InstanceNorm2d(num_channel, affine=True),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            # nn.InstanceNorm2d(num_channel, affine=True),
            LayerNorm(num_channel),
            nn.PReLU(num_channel),
        )
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv(x)
        return x


class AFN_HearNet(nn.Module):
    def __init__(
        self, nframe: int, nhop: int, mid_channel: int = 64, conformer_num=4, fs=16000
    ) -> None:
        super().__init__()

        self.stft = STFT(nframe, nhop, nframe)
        self.reso = fs / nframe
        nbin = nframe // 2 + 1
        assert 250 % self.reso == 0
        self.freqs = torch.linspace(0, fs // 2, nbin)  # []

        # self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.preprocess = HLModule(nbin, HL_freq_extend=self.freqs)
        self.group = mid_channel
        # self.cbook = GumbelVectorQuantizer(nbin, 128, self.group, 65 * self.group)

        self.mlp = nn.Sequential(
            nn.Linear(nbin, nbin * 4),
            Rearrange("b c t (f n)-> b (c n) t f", n=4),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(4, 16, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 64, (1, 3), (1, 2), (0, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            Rearrange("b c t f-> b f t c"),
            nn.Linear(64, mid_channel),
        )
        # self.hl_attn = HLAttn()

        self.encoder = DenseEncoder(in_channel=2, channels=mid_channel)

        self.conformer = nn.ModuleList(
            [AMFTConformer(dim=mid_channel) for _ in range(conformer_num)]
        )

        self.mask_decoder = MaskDecoder(
            num_features=nbin, num_channel=mid_channel, out_channel=1
        )
        self.complex_decoder = ComplexDecoder(num_channel=mid_channel)

        # self.cbook = nn.ModuleList(
        #     [CodeBook(num_cb=64, dim_cb=65, mid_channel=4, dim_inp=16) for _ in range(2)]
        # )
        # self.factAttn = FactorizedAttn(1, 16, 8)
        self.vad_predictor = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 65), stride=(1, 65)),  # B,C,T,1
            nn.Conv2d(
                in_channels=mid_channel,
                out_channels=mid_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ),  # b,c,t,1
            Rearrange("b c t ()->b t c"),
            nn.LayerNorm(mid_channel),
            nn.PReLU(),
            nn.GRU(
                input_size=mid_channel,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
            ),
        )

        self.vad_post = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, HL):
        """
        x: B,T
        HL: B,6
        """
        xk = self.stft.transform(x)
        xk_mag = xk.pow(2).sum(1, keepdim=True).sqrt()  # B,1,T,F
        # xk_bands_pow = compute_subbands_energy(xk, self.ht_freq)

        # xk_b: b,t,16; hl_b: b,t,16
        hl_b = self.preprocess.extend_with_linear(HL)  # b,1,1,f
        hl_b = self.mlp(hl_b)  # b,1,1,f
        # hl_b = hl_b.view(-1, hl_b.size(-1))  # b,mch
        hl_b = hl_b.squeeze(2)

        # b,16,t,f
        # xk_hl, _ = self.hl_attn(xk, hl_b)

        xk = self.encoder(xk)

        for l in self.conformer:
            xk, _ = l(xk, hl_b, causal=True)

        vad_pred, _ = self.vad_predictor(xk)
        vad = self.vad_post(vad_pred)

        # x_fact = self.factAttn(xk, x_hl)
        # xk: b,c,t,f; x_hl: b,c,t,1
        mask = self.mask_decoder(xk)
        spec = self.complex_decoder(xk)
        r, i = spec.chunk(2, dim=1)  # b,1,t,f
        phase = torch.atan2(i, r)

        # mask = self.factAttn(mask, xk_hl)
        xk_mag_est = xk_mag * mask
        spec_r = xk_mag_est * torch.cos(phase)
        spec_i = xk_mag_est * torch.sin(phase)

        xk_est = torch.concat([spec_r, spec_i], dim=1)

        x = self.stft.inverse(xk_est)

        return x, vad

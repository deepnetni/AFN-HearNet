import torch
from model.check_flops import check_flops
from model.AFN_HearNet import AFN_HearNet


if __name__ == "__main__":
    x = torch.randn(1, 16000)
    hl = torch.randn(1, 6)

    net = AFN_HearNet(512, 256, 48, 2)
    net.eval()
    flops, params = check_flops(net, x, hl)

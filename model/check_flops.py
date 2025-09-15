import warnings
from thop import profile
from ptflops import get_model_complexity_info


def check_flops(net, *args):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This API is being deprecated")
        flops, params = profile(net, inputs=(*args,), verbose=False)
    print(f"FLOPs={flops / 1e9:.2f} G/s, params={params/1e6:.2f} M")
    return flops, params


def check_each_layer(net, inp):
    """
    inp: shape, only supported single input, (B,I)
    """
    flops, params = get_model_complexity_info(
        net, (*inp,), as_strings=True, print_per_layer_stat=True
    )
    print(flops)
    print(params)

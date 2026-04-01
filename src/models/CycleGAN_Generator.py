import torch
import torch.nn as nn
from torch.nn import init
import functools

# NOTE!!! This code is an adapted version of the CycleGAN repo, we just adapted it to our needs but have no rights over it´s design
# Original code from:https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

# Helper functions
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type="normal", init_gain=0.02):
    if torch.cuda.is_available():
        net.to(0)
        print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net

# Generator & Discriminator
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect": conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate": conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero": p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout: conv_block += [nn.Dropout(0.5)]
        if padding_type == "reflect": conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate": conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero": p = 1
        conv_block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type="reflect"):
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial: use_bias = norm_layer.func == nn.InstanceNorm2d
        else: use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, 7, 1, 0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, 3, 2, 1, bias=use_bias), norm_layer(ngf*mult*2), nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_blocks): model += [ResnetBlock(ngf*mult, padding_type, norm_layer, use_dropout, use_bias)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), 3, 2, 1, output_padding=1, bias=use_bias), norm_layer(int(ngf*mult/2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7, 1, 0), nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        return self.model(input)

def define_G(input_nc, output_nc, ngf, netG, norm="instance", use_dropout=False, init_type="normal", init_gain=0.02):
    norm_layer = get_norm_layer(norm)
    if netG == "resnet_9blocks": net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks": net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else: raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return init_net(net, init_type, init_gain)


def define_G_inference(input_nc, output_nc, ngf, netG, weights_path=None, norm="instance", use_dropout=False, init_type="normal", init_gain=0.02):
    """
    Returns a generator ready for inference. 
    If weights_path is provided, loads the .pth/.pt file.
    """
    net = define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain)
    
    if weights_path is not None:
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        # Strip 'module.' if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        net.load_state_dict(new_state_dict, strict=True)
    
    net.eval()  # inference mode
    return net
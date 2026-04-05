import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np

# NOTE: This code is adapted from the Contrastive Unpaired Translation (CUT) repository
# Original code from: https://github.com/taesungp/contrastive-unpaired-translation


#Helper functions -------------- 
def get_filter(filt_size=3):
    """Get antialiasing filter for downsampling/upsampling."""
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    """Get padding layer."""
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample(nn.Module):
    """Antialiased downsampling layer."""
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), 
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    """Antialiased upsampling layer."""
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer."""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: register device and initialize weights."""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    init_weights(net, init_type, init_gain=init_gain)
    return net


# Generator -------------

class ResnetBlock(nn.Module):
    """Define a Resnet block."""
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block."""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """Resnet-based generator from CUT with optional antialiasing."""
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False):
        """
        Construct a Resnet-based generator.
        
        Parameters:
            input_nc (int) -- number of input channels
            output_nc (int) -- number of output channels
            ngf (int) -- number of filters in the last conv layer
            norm_layer -- normalization layer
            use_dropout (bool) -- if use dropout layers
            n_blocks (int) -- number of ResNet blocks
            padding_type (str) -- padding layer type: reflect | replicate | zero
            no_antialias (bool) -- if True, use stride 2 convs instead of antialiased downsampling
            no_antialias_up (bool) -- if True, use transposed convs instead of antialiased upsampling
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward pass."""
        return self.model(input)


# Inference

def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, 
             init_type='normal', init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[]):
    """
    Create a CUT generator.
    
    Parameters:
        input_nc (int) -- number of input image channels
        output_nc (int) -- number of output image channels
        ngf (int) -- number of generator filters in the last conv layer
        netG (str) -- architecture name: resnet_9blocks | resnet_6blocks
        norm (str) -- normalization layer type: batch | instance | none
        use_dropout (bool) -- if use dropout layers
        init_type (str) -- initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for initialization
        no_antialias (bool) -- if True, use stride 2 convs (default: False, use antialiasing)
        no_antialias_up (bool) -- if True, use transposed convs (default: False, use antialiasing)
        gpu_ids (int list) -- which GPUs to use; e.g., [0] or [0,1,2]
    
    Returns:
        A generator network
    """
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, 
                              no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_G_inference(input_nc, output_nc, ngf, netG, weights_path=None, norm='instance', 
                       use_dropout=False, init_type='normal', init_gain=0.02, 
                       no_antialias=False, no_antialias_up=False):
    """
    Create a CUT generator ready for inference.
    
    This function builds the generator and optionally loads pretrained weights.
    For inference, set no_antialias and no_antialias_up to match your training configuration.
    
    Parameters:
        input_nc (int) -- number of input image channels (typically 3 for RGB)
        output_nc (int) -- number of output image channels (typically 3 for RGB)
        ngf (int) -- number of generator filters (typically 64)
        netG (str) -- architecture: 'resnet_9blocks' or 'resnet_6blocks'
        weights_path (str) -- path to .pth checkpoint file (e.g., 'latest_net_G.pth')
        norm (str) -- normalization: 'instance' (default for CUT), 'batch', or 'none'
        use_dropout (bool) -- if dropout was used during training (typically False)
        init_type (str) -- initialization method (only used if weights_path is None)
        init_gain (float) -- initialization gain (only used if weights_path is None)
        no_antialias (bool) -- must match training; default False means antialiasing was used
        no_antialias_up (bool) -- must match training; default False means antialiasing was used
    
    Returns:
        A generator network in eval mode, ready for inference
    
    Example:
        gen = define_G_inference(3, 3, 64, 'resnet_9blocks', 
                                 weights_path='/path/to/latest_net_G.pth',
                                 norm='instance', no_antialias=False, no_antialias_up=False)
        gen.eval()
        with torch.no_grad():
            fake_B = gen(real_A)
    """
    net = define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain, 
                   no_antialias, no_antialias_up, gpu_ids=[])
    
    if weights_path is not None:
        print(f"Loading CUT generator weights from {weights_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(weights_path, map_location=device)
        
        # Strip 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        net.load_state_dict(new_state_dict, strict=True)
        print("Weights loaded successfully")
    
    net.eval()
    return net

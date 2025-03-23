import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from IPython import embed
import lpips  # 即使不使用，代码中已经导入，可以暂时保留
from tqdm import tqdm
from ema_pytorch import EMA
import cv2
import random
from datetime import datetime
from shutil import get_terminal_size
import math
import yaml
import os

# 移除重复导入的 logging
logger = logging.getLogger("base")

yaml_config_string = """
name: refusion
suffix: "testresult" # add suffix to saved images
model: denoising
gpu_ids: [0]
results_root: testresult
sde:
  max_sigma: 70
  T: 1000

degradation:
  sigma: 50
  noise_type: G # Gaussian noise: G

datasets:
  test1:
    name: NTIRETEST
    mode: LQ
    dataroot_LQ: /root/autodl-tmp/image-restoration-sde/dataset/LSDIR/test
    dataroot_GT: None

#### network structures
network_G:
  which_model_G: ConditionalNAFNet
  setting:
    width: 64
    enc_blk_nums: [1, 1, 1, 28]
    middle_blk_num: 1
    dec_blk_nums: [1, 1, 1, 1]

#### path
path:
  pretrain_model_G: log/refusion/models/700000_G.pth
  strict_load: true
"""

# 使用 yaml.safe_load() 解析 YAML 字符串
opt = yaml.safe_load(yaml_config_string)
opt['is_train'] = False # 手动添加 'is_train' 键

def OrderedYaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    yaml.Dumper.add_representer(OrderedDict, dict_representer)
    yaml.Loader.add_constructor(_mapping_tag, dict_constructor)
    return yaml.Loader, yaml.Dumper


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + "_archived_" + get_timestamp()
        print("Path already exists. Rename it to [{:s}]".format(new_name))
        logger = logging.getLogger("base")
        logger.info("Path already exists. Rename it to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


class ProgressBar(object):
    """A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                "terminal width is too small ({}), please consider widen the terminal for better "
                "progressbar visualization".format(terminal_width)
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(
                "[{}] 0/{}, elapsed: 0s, ETA:\n{}\n".format(
                    " " * self.bar_width, self.task_num, "Start..."
                )
            )
        else:
            sys.stdout.write("completed: 0, elapsed: 0s")
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg="In progress..."):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / self.task_num
            eta = int((1 - percentage) * elapsed / percentage + 0.5) if percentage > 0 else 0
            mark_width = int(self.bar_width * percentage)
            arrow = ">" if mark_width > 0 else ""
            padding = " " * (self.bar_width - mark_width)
            msg = "\r[{}{}] {}/{}, {:.2f} task/s, elapsed: {}s, ETA: {:d}s".format(
                "#" * mark_width, padding + arrow, self.completed, self.task_num, fps, int(elapsed + 0.5), eta
            )
        else:
            msg = "\rcompleted: {}, elapsed: {:.2f}s, {:.2f} tasks/s".format(
                self.completed, elapsed, fps
            )
        sys.stdout.write(msg + msg[-1])
        sys.stdout.flush()


#################### options ####################
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YAML file.")
    parser.add_argument(
        "--launcher",
        default="none",
        type=str,
        help="job launcher for distributed training",
        choices=["none", "pytorch"],
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    return args


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * indent_l + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * indent_l + "]\n"
        else:
            msg += " " * indent_l + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_id):
    logger = logging.getLogger("base")
    phase = opt["phase"]
    if resume_id is None:
        logger.warning(
            "Warning: not specified resume_id, so skip check resume."
        )
        return None
    else:
        resume_path = os.path.join(opt["path"]["root"], phase, resume_id)
        logger.info("Checking resume path: {}".format(resume_path))
        if os.path.exists(os.path.join(resume_path, "{}_opt.json".format(phase))):
            return resume_path
        else:
            logger.warning(
                "Warning: resume_id ({}) is not valid. Skip check resume.".format(
                    resume_id
                )
            )
            return None


#################### models ####################
class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.is_train = opt["is_train"]

    def feed_data(self, data):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def load(self):
        pass

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith("module."):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        network.load_state_dict(load_net_clean, strict=strict)

# networks.py 内容
import logging
import torch

logger = logging.getLogger("base")
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch import einsum

from einops import rearrange, reduce
from einops.layers.torch import Rearrange


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape)) # 将 weight 更名为 gamma
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 将 bias 更名为 beta
        self.g = nn.Parameter(torch.ones(normalized_shape))     # 保留 g 参数
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                f"data_format not in ['channels_last', 'channels_first'] got {self.data_format}"
            )
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.gamma * self.g, self.beta, self.eps) # 应用 gamma 和 g
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = (self.gamma * self.g)[:, None, None] * x + self.beta[:, None, None] # 应用 gamma 和 g
            return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size//2), bias=bias)


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = nn.Sequential(
            act, nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# channel attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


# self attention on each channel
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time


class ConditionalNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x, time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(x.device)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)

        x = x[..., :H, :W]

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x




class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, act_layer=nn.Identity, norm_layer=nn.BatchNorm2d,
                 drop_rate=0., num_branches=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_branches = num_branches

        self.branches = nn.ModuleList()
        for k in range(num_branches):
            kernel_size = 2 * k + 3
            padding = kernel_size // 2
            self.branches.append(ConvLayer(self.in_channels, self.out_channels, kernel_size=kernel_size, padding=padding))

        self.norm = norm_layer(self.out_channels)
        self.act = act_layer()
        self.drop = nn.Dropout(drop_rate)

        if self.in_channels != self.out_channels:
            self.proj = ConvLayer(self.in_channels, self.out_channels, kernel_size=1, padding=0)
        else:
            self.proj = nn.Identity()

    def forward(self, x, emb=None):
        res = self.proj(x)
        out = 0
        for branch in self.branches:
            out += branch(x)
        out = self.norm(out)
        if emb is not None:
            out = out + emb.view(emb.shape[0], -1, 1, 1)
        out = self.act(out)
        out = self.drop(out)
        return out + res
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, act='silu', norm='ln', bn_weight_init=1): # Changed default norm to 'ln'
        super().__init__()
        if in_channels is None:
            self.add_module('conv', nn.Conv2d(3, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        elif out_channels is None:
            self.add_module('conv', nn.Conv2d(in_channels, 3, kernel_size, stride, padding, dilation, groups, bias))
        else:
            self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        if norm == 'bn':
            self.add_module('norm', nn.BatchNorm2d(out_channels, affine=True))
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)
        elif norm == 'ln':
            self.add_module('norm', LayerNorm(out_channels))
        if act == 'silu':
            self.add_module('act', nn.SiLU())
        elif act == 'relu':
            self.add_module('act', nn.ReLU(inplace=True))
        elif act == 'gelu':
            self.add_module('act', nn.GELU())
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = nn.Parameter(torch.ones(normalized_shape)) # Add scale parameter
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                f"data_format not in ['channels_last', 'channels_first'] got {self.data_format}"
            )
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight * self.scale, self.bias, self.eps) # Apply scale
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = (self.weight * self.scale)[:, None, None] * x + self.bias[:, None, None] # Apply scale
            return x

# lr_scheduler.py 内容
import math
from collections import Counter, defaultdict

import torch
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepLR_Restart(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        restarts=None,
        weights=None,
        gamma=0.1,
        clear_state=False,
        last_epoch=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.gamma_ = 0.5
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            # print(self.optimizer.param_groups)
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma_ ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(
        self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1
    ):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (
            2 * self.T_max
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max))
            / (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

# optimizer.py 内容
# Copyright 2023 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch implementation of the Lion optimizer."""

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.

        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
          lr (float, optional): learning rate (default: 1e-4)
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.99))
          weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.

        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

# loss.py 内容
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()

class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.define_network(opt)
        self.load()  # load pretrained models

    def init_print(self):
        """Print the network and model information"""
        opt = self.opt
        phase = opt["phase"]
        logger = logging.getLogger("base")
        print_network, print_params = opt.get("print_network", False), opt.get(
            "print_params", False
        )
        if print_network:
            self.print_network()
        if print_params:
            n_params = self.get_network_description(self.netG)
            logger.info(
                "Number of G params: {:,d}".format(n_params[1])
            )  # print G number of parameters

    def define_network(self, opt):
        """Create netG"""
        gpu_ids = opt["gpu_ids"]
        opt_net = opt["network_G"]
        which_model = opt_net["which_model_G"]

        if which_model == "ConditionalNAFNet":
            net_G = ConditionalNAFNet(**opt_net["setting"])
        else:
            raise NotImplementedError(
                "Network [{:s}] is not found".format(which_model)
            )

        if gpu_ids:
            assert torch.cuda.is_available()
            netG = net_G.to(self.device)
            if len(gpu_ids) > 1:
                netG = nn.DataParallel(netG)
        return netG

    import os

    def load(self):
        """Load pretrained model for G"""
        load_path_G = self.opt["path"].get("pretrain_model_G", None)
        if load_path_G is not None:
            logger = logging.getLogger("base")
            logger.info(
                "Loading model for G [{:s}] ...".format(load_path_G)
            )
            try:
                pretrained_dict = torch.load(load_path_G, map_location=self.device)
                output_file = "pretrained_model_keys.txt"
                with open(output_file, "w") as f:
                    for key in pretrained_dict.keys():
                        f.write(key + "\n")
                # print(f"预训练模型 state_dict 的键已保存到文件: {output_file}")
                self.load_network(load_path_G, self.netG, self.opt["path"].get("strict_load", True))
            except Exception as e:
                print(f"加载预训练模型时发生错误: {e}")
    def test(self):
        """Testing function.

        Args:
            path (str): Path to the LQ images.

        Returns:
            save_results (dict): Ordered dictionary of result paths for each image.
        """
        self.netG.eval()
        with torch.no_grad():
            LQ = self.var_LQ
            noise_level = self.var_noise_level
            sigma = noise_level.float().view(-1, 1, 1, 1)
            x_noisy = LQ + torch.randn_like(LQ) * sigma
            self.output = self.netG(x_noisy, sigma.squeeze(-1))
        self.netG.train()

    def feed_data(self, data, need_GT=True):
        self.var_LQ = data["LQ"].to(self.device)
        if need_GT:
            self.var_GT = data["GT"].to(self.device)
        self.var_noise_level = data["noise_level"].to(self.device)

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_LQ.detach().cpu()
        out_dict["rlt"] = self.output.detach().cpu()
        if need_GT:
            out_dict["GT"] = self.var_GT.detach().cpu()
        return out_dict

    def get_current_losses(self):
        return OrderedDict()

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = "Network G (input: {}, output: {}):\n".format(
                self.opt["datasets"]["train"]["LQ_size"],
                self.opt["datasets"]["train"]["GT_size"],
            )
        else:
            net_struc_str = "Network G (input: {}, output: {}):\n".format(
                self.opt["datasets"]["train"]["LQ_size"],
                self.opt["datasets"]["train"]["GT_size"],
            )
        net_struc_str += s
        logger.info(net_struc_str)

# 现在实例化模型并访问参数
if __name__ == "__main__":
    # 打印整个配置
    print("Full Configuration:")
    print(opt)
    print("-" * 40)
    # 访问特定的配置参数
    print("Specific Configuration Parameters:")
    print(f"Model Name: {opt['name']}")
    print(f"Suffix: {opt['suffix']}")
    print(f"Model Type: {opt['model']}")
    print(f"GPU IDs: {opt['gpu_ids']}")
    print(f"Results Root: {opt['results_root']}")
    print("-" * 40)

    print("SDE Configuration:")
    print(f"Max Sigma: {opt['sde']['max_sigma']}")
    print(f"T: {opt['sde']['T']}")
    print("-" * 40)

    print("Degradation Configuration:")
    print(f"Sigma: {opt['degradation']['sigma']}")
    print(f"Noise Type: {opt['degradation']['noise_type']}")
    print("-" * 40)

    print("Network G Configuration:")
    print(f"Which Model G: {opt['network_G']['which_model_G']}")
    print(f"Network G Setting: {opt['network_G']['setting']}")
    print("-" * 40)

    print("Path Configuration:")
    print(f"Pretrained Model G Path: {opt['path']['pretrain_model_G']}")
    print(f"Strict Load: {opt['path']['strict_load']}")
    print("-" * 40)

    # 实例化模型
    try:
        model = DenoisingModel(opt)
        print("\nModel instantiated successfully.")

        # 你可以进一步访问模型的属性，例如网络G
        print("\nNetwork G:")
        print(model.netG)

    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("请确保所有必要的模型定义都已包含在此脚本中。")
    except Exception as e:
        print(f"\nAn error occurred during model instantiation: {e}")
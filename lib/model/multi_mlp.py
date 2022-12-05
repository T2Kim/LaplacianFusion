'''Define basic blocks
'''

import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

'''Adapted from the SIREN repository https://github.com/vsitzmann/siren
'''

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init,last_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None,None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None,None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None,None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None,None),
                         'softplus':(nn.Softplus(), init_weights_normal, None,None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None,None)}

        nl, nl_weight_init, first_layer_init,last_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(nn.Linear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

        if last_layer_init is not None:
            self.net[-1].apply(last_layer_init)

    def forward(self, coords, **kwargs):
        output = self.net(coords)
        return output

def linear_pack(linears, x):
    # x    (B, F_in) or (N, B, F_in)
    # w    (F_in, F_out)
    N = len(linears)
    F_out = linears[0].weight.shape[0]

    # bw    (N, F_in, F_out)
    bw = torch.stack([l.weight.T for l in linears])
    # bb    (N, F_out)
    bb = torch.stack([l.bias for l in linears]).view(N, 1, F_out)

    # (N, B, F_out)
    return torch.matmul(x, bw) + bb

class MultiLinear(nn.Module):
    def __init__(self, num_mlp, in_features, out_features):
        super().__init__()
        self.N = num_mlp
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((self.N, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((self.N, out_features)))
        self.reset_parameters()

    def forward(self, x):
        # aaa = x.reshape(-1, self.N, self.in_features).permute(1, 0, 2)
        # ddd = self.weight.permute(0, 2, 1)
        # dsd = self.bias.view(self.N, 1, -1)
        # ssa = torch.matmul(self.weight, x.unsqueeze(-1))
        if len(x.shape) == 2:
            return (torch.matmul(self.weight, x.unsqueeze(-1)).squeeze() + self.bias)
        else:
            return torch.matmul(x, self.weight.permute(0, 2, 1)) + self.bias.view(self.N, 1, -1)
        # return (torch.matmul(x, self.weight.permute(0, 2, 1)) + self.bias.view(self.N, 1, -1)).reshape(-1, self.out_features)
        # return (torch.matmul(x.reshape(-1, self.N, self.in_features).permute(1, 0, 2), self.weight.permute(0, 2, 1)) + self.bias.view(self.N, 1, -1)).permute(1, 0, 2).reshape(-1, self.out_features)
        # return torch.matmul(x, self.weight.permute(0, 2, 1)) + self.bias.view(self.N, 1, -1)
        # tmp_b = x.shape[0] // self.weight.shape[0]
        # return torch.bmm(self.weight.unsqueeze(0).repeat(tmp_b, 1, 1, 1).view(x.shape[0], self.out_features, self.in_features), x.unsqueeze(-1)).squeeze() + \
            # self.bias.unsqueeze(0).repeat(tmp_b, 1, 1).view(x.shape[0], self.out_features)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

class MultiFCBlock(nn.Module):

    def __init__(self, num_mlp, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()
        self.N = num_mlp

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init,last_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None,None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None,None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None,None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None,None),
                         'softplus':(nn.Softplus(), init_weights_normal, None,None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None,None)}

        nl, nl_weight_init, first_layer_init,last_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            MultiLinear(self.N, in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                MultiLinear(self.N, hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(MultiLinear(self.N, hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                MultiLinear(self.N, hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

        if last_layer_init is not None:
            self.net[-1].apply(last_layer_init)

    def forward(self, coords, **kwargs):
        x = self.net(coords)
        return x

class LinearNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)

    def forward(self, x):
        return self.net(x)

class MultiLinearNet(nn.Module):
    def __init__(self, num_mlp, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = MultiFCBlock(num_mlp, in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)

    def forward(self, x):
        return self.net(x)


def init_weights_normal(m):
    if type(m) == MultiLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_selu(m):
    if type(m) == MultiLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == MultiLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == MultiLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def last_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchmeta.modules import (MetaModule, MetaSequential)
# from torchmeta.modules.utils import get_subdict
from lib.model.embedder import *
from collections import OrderedDict

import numpy as np


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
    

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


class MLP(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 600, num_hidden_layer = 5, output_dim = 3):
        super(MLP, self).__init__()

        self.fc_input = nn.Linear(manifold_pos_dim + pose_dim, hidden_dim)
        # self.fc_input = nn.Linear(manifold_pos_dim, hidden_dim)

        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)

        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            layers.append(nn.ReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        x = self.fc_input(torch.cat((input_manifold_pos, input_pose_code), dim=1))
        # x = self.fc_input(input_manifold_pos)
        x = self.fc_hidden(x)
        return self.output_linear(x)

class MLP_CDF(nn.Module):
    # def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 1000, num_hidden_layer = 8, output_dim = 3):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 400, num_hidden_layer = 5, output_dim = 3):
        super(MLP_CDF, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim + pose_dim, hidden_dim)
        # self.fc_input = nn.Linear(manifold_pos_dim, hidden_dim)

        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)

        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.output_lapl_b2 = nn.Parameter(torch.ones(hidden_dim) * 10)
        self.output_lapl_b = nn.Parameter(torch.ones(output_dim) * 10)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            # layers.append(Sine())
            # layers.append(nn.ReLU())
            layers.append(nn.LeakyReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward1(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        # return self.output_linear(x)

        x = self.output_linear(x)
        x = F.tanh(x)
        # x = torch.clamp(x, min=-1, max=1)
        # return -10 * torch.sign(x) * torch.log(1 - torch.abs(x) + 1e-16)
        return -self.output_lapl_b.unsqueeze(0) * torch.sign(x) * torch.log(1 - torch.abs(x) + 1e-16)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        # return self.output_linear(x)

        # x = self.output_linear(x)
        x = F.tanh(x)
        # x = torch.clamp(x, min=-1, max=1)
        # return -10 * torch.sign(x) * torch.log(1 - torch.abs(x) + 1e-16)
        return self.output_linear(-self.output_lapl_b2.unsqueeze(0) * torch.sign(x) * torch.log(1 - torch.abs(x) + 1e-16))


class MLP_CDF_lin(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 1000, num_hidden_layer = 4, output_dim = 3):
        super(MLP_CDF_lin, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(8, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim + pose_dim, hidden_dim)
        # self.fc_input = nn.Linear(manifold_pos_dim, hidden_dim)

        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)

        self.output_linear = nn.Linear(hidden_dim, output_dim)
        # self.output_lapl_b = nn.Parameter(torch.ones(hidden_dim) * 10)
        self.output_lapl_b = nn.Parameter(torch.ones(output_dim) * 10)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            # layers.append(Sine())
            # layers.append(nn.ReLU())
            layers.append(nn.LeakyReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        return self.output_linear(x)



class MLP_Detail(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 800, num_hidden_layer = 3, output_dim = 3):
        super(MLP_Detail, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)
        # self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim + pose_dim, hidden_dim)
        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            layers.append(nn.LeakyReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        return self.output_linear(x)


class MLP_Coarse_res(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 600, num_hidden_layer = 5, output_dim = 3):
        super(MLP_Coarse_res, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim + pose_dim, hidden_dim)

        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)

        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def make_layer(self, layer, num):
        layers = []
        for _ in range(num):
            # layers.append(Sine())
            # layers.append(nn.ReLU())
            layers.append(nn.LeakyReLU())
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(torch.cat((input_pos_embed, input_pose_code), dim=1))
        x = self.fc_hidden(x)
        return self.output_linear(x)


class MLP_Coarse_res2(nn.Module):
    def __init__(self, manifold_pos_dim, pose_dim, hidden_dim = 600, num_hidden_layer = 5, output_dim = 3):
        super(MLP_Coarse_res2, self).__init__()

        self.pos_embedder, pos_embedder_out_dim = get_embedder_nerf(10, input_dims=manifold_pos_dim, i=0)

        self.fc_input = nn.Linear(pos_embedder_out_dim, hidden_dim)
        # self.fc_input = nn.Linear(2, hidden_dim)

        self.fc_hidden = self.make_layer(nn.Linear(hidden_dim, hidden_dim), num_hidden_layer)

        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def make_layer(self, layer, num):
        layers = []
        # layers.append(nn.Softplus())
        # layers.append(nn.LeakyReLU())
        for _ in range(num):
            layers.append(layer)
            layers.append(nn.LeakyReLU())
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Softplus())
        return nn.Sequential(*layers)

    def forward(self, input_manifold_pos, input_pose_code):
        input_pos_embed = self.pos_embedder(input_manifold_pos)
        x = self.fc_input(input_pos_embed)
        x = self.fc_hidden(x)
        return self.output_linear(x)


import torch
from torch import nn
import numpy as np
import math
# import tinycudann as tcnn


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


class Homography(nn.Module):
    def __init__(self, in_features=1, hidden_features=256, hidden_layers=1):
        super().__init__()
        out_features = 8
        
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.Linear(hidden_features, out_features))     
        self.net = nn.Sequential(*self.net)
        
        self.init_weights()
        
    def init_weights(self):
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.Tensor([1., 0., 0., 0., 1., 0., 0., 0.]))
    
    def forward(self, coords):
        output = self.net(coords)
        return output
    

class Annealed(nn.Module):
    def __init__(self, in_channels, annealed_step, annealed_begin_step=0, identity=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Annealed, self).__init__()
        self.N_freqs = 16
        self.in_channels = in_channels
        self.annealed = True
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step

        self.index = torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.identity = identity

        self.index_2 = self.index.view(-1, 1).repeat(1, 2).view(-1)

    def forward(self, x_embed, step):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        use_PE = False

        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = (self.N_freqs) * (step - self.annealed_begin_step) / float(
                    self.annealed_step)

        w = (1 - torch.cos(math.pi * torch.clamp(alpha * torch.ones_like(self.index_2) - self.index_2, 0, 1))) / 2
        
        if use_PE:
            w[16:] = w[:16]

        out = x_embed * w.to(x_embed.device)

        return out
    

class BARF_PE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=2,
                                     encoding_config=config["positional encoding"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims +
                                    2,
                                    n_output_dims=3,
                                    network_config=config["BARF network"])

    def forward(self, x, step=0, aneal_func=None):
        input = x
        input = self.encoder(input)
        if aneal_func is not None:
            input = torch.cat([x, aneal_func(input,step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)
        
        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input)
        return x
    

class Deform_Hash3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding_deform3d"])
        self.decoder = nn.Sequential(nn.Linear(self.encoder.n_output_dims + 3, 256),
                                     nn.ReLU(), 
                                     nn.Linear(256, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, 256), 
                                     nn.ReLU(),
                                     nn.Linear(256, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 2)
                                     )

    def forward(self, x, step=0, aneal_func=None):
        input = x
        input = self.encoder(input)
        if aneal_func is not None:
            input = torch.cat([x, aneal_func(input,step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)

        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input) / 5

        return x


class Deform_Hash3d_Warp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Deform_Hash3d = Deform_Hash3d(config)

    def forward(self, xyt_norm, step=0,aneal_func=None):
        x = self.Deform_Hash3d(xyt_norm,step=step, aneal_func=aneal_func)

        return x
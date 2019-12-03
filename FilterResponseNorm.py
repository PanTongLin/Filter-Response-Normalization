import torch
import torch.nn as nn


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, use_TLU=True):
        super(FilterResponseNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.use_TLU = use_TLU

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        if use_TLU:
            self.tau = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('tau', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if self.use_TLU:
            nn.init.zeros_(self.tau)

    def forward(self, input):

        nu2 = torch.mean(input.pow(2), (2, 3), keepdim=True)

        out = input * torch.rsqrt(nu2 + abs(self.eps))
        weight = self.weight[..., None, None]
        bias = self.bias[..., None, None]
        # Return after applying the Offset-ReLU non-linearity
        if self.use_TLU:
            tau = self.tau[..., None, None]
            return torch.max(weight*out + bias, tau)
        else:
            return self.gamma*out + self.bias

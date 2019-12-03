import torch
import torch.nn as nn


class FilterResponseNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, momentum=0.1, use_TLU=True):
        super(FilterResponseNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.use_TLU = use_TLU

        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        if use_TLU:
            self.tau = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('tau', None)

        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_running_stats(self):
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / \
                    float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            nu2 = torch.mean(input.pow(2), (2, 3), keepdim=True)
            self.running_var = (1-exponential_average_factor) * \
                self.running_var + exponential_average_factor*nu2

        out = input * torch.rsqrt(self.running_var + abs(self.eps))
        # Return after applying the Offset-ReLU non-linearity
        if self.use_TLU:
            return torch.max(self.weight*out + self.bias, self.tau)
        else:
            return self.gamma*out + self.bias

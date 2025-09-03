import torch.nn as nn
from src.models.model_utils import ClampSTE, clampSTE_max


class BNNet(nn.Module):
    def __init__(self, input_size=1, output_size=1, hl_size=100, **kwargs):
        super(BNNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hl_size),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hl_size, hl_size),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hl_size, output_size),)

    def forward(self, x):
        x = self.model(x)
        return ClampSTE.apply(x)


class ClampLRNet(nn.Module):
    def __init__(self, input_size=1, num_hl=1, hl_size=100, output_size=1, max_limit=1., **kwargs):
        super(ClampLRNet, self).__init__()
        self.num_hl = num_hl
        self.first_layer = nn.Sequential(nn.Linear(input_size, hl_size),
                                         nn.ReLU(inplace=True))

        for idx in range(num_hl):
            setattr(self, f'hl_{idx}', nn.Sequential(
                nn.Linear(hl_size, hl_size), nn.ReLU(inplace=True)))

        self.last_layer = nn.Linear(hl_size, output_size)
        self.max_limit = max_limit

    def forward(self, x):
        x = self.first_layer(x)
        for idx in range(self.num_hl):
            x = getattr(self, f'hl_{idx}')(x)
        x = self.last_layer(x)
        return clampSTE_max(x, max_limit=self.max_limit)


class BTNet(nn.Module):
    def __init__(self, input_size=1, num_hl=1, hl_size=100, output_size=1, max_limit=1, **kwargs):
        super(BTNet, self).__init__()
        self.num_hl = num_hl
        self.first_layer = nn.Sequential(nn.Linear(input_size, hl_size),
                                         nn.PReLU())
        self.last_layer = nn.Linear(hl_size, output_size)
        self.max_limit = max_limit

    def forward(self, x):
        x = self.first_layer(x)
        x = self.last_layer(x)
        return clampSTE_max(x, min_limit=1e-4, max_limit=self.max_limit)


def weights_init(m, g, b):
    if m.__class__.__name__.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight, g)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(b)

import torch.nn as nn
"""
# 1. **ResBlock** : A basic block with residual connections, which contains a two-layer fully connected network
#   and ReLU activation function, and adds the input to the output via residual connections.
# 2. **noResBlock** : A basic block without residual connections, containing a three-layer fully connected network
#   and ReLU activation function.
# 3. **linearNet** : Stacks multiple 'ResBlocks' or' NoResBlocks' on top of each other to form a multi-layer network.
# The network can create deep models by setting different number of blocks (' num_blocks').
# 4. **Encoder** : The encoder model, which uses' ResBlock 'as the basic module, processes the input data
#   and maps it to a low-dimensional feature space.
# 5. **Decoder** : The decoder model, which uses' noResBlock 'as the basic module to remap the output of the encoder
#   back to the original input dimension.

"""

class ResBlock(nn.Module):
    def __init__(self, input_dim, feature_dim,mid_dim):
        super(ResBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, feature_dim),
            nn.ReLU()
        )
        self.residual = nn.Linear(input_dim, feature_dim)

    def forward(self, x):
        y = self.encoder(x)
        return y+self.residual(x)


class noResBlock(nn.Module):
    def __init__(self, input_dim, feature_dim,mid_dim):
        super(noResBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, feature_dim),
        )

    def forward(self, x):
        y = self.decoder(x)
        return y


class linearNet(nn.Module):
    def __init__(self, block, infeature, outfeature,mid_dim, num_blocks):
        super(linearNet, self).__init__()
        self.in_channels = infeature
        self.layer = self.make_layer(block, outfeature,mid_dim, num_blocks)

    def make_layer(self, block, out_channels, mid_dim,num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_channels, out_channels,mid_dim),)
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim,mid_dim,layers):
        super(Encoder, self).__init__()
        self.encoder = linearNet(ResBlock, input_dim, feature_dim,mid_dim, layers)

    def forward(self, x):
        y = self.encoder(x)
        return y


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim,mid_dim,layers):
        super(Decoder, self).__init__()
        self.decoder = linearNet(noResBlock, feature_dim,input_dim,mid_dim, layers)

    def forward(self, x):
        y = self.decoder(x)
        return y



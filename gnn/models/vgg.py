import torch
import torch.nn as nn

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class VGG(nn.Module):
    def __init__(self, dim, channel):
        super(VGG, self).__init__()
        self.dim = dim
        # 18 x 18
        self.c1 = nn.Sequential(
                vgg_layer(channel, 64),
                vgg_layer(64, 64),
                )
        # 9 x 9
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 4 x 4 
        self.c3 = nn.Sequential(
                nn.Conv2d(128, dim, 4, 1, 0),
                nn.Tanh()
                )
        
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.c1(input) 
        h2 = self.c2(self.mp(h1))
        h3 = self.c3(self.mp(h2))
        return h3
        # return h5.view(-1, self.dim)
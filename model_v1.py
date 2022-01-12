import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# In original paper the output shape is not similar to the input shape
# Here we are going to make both similar that's why I decided to put padding in Convolutional layer

class ConvCat(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(ConvCat, self).__init__()
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ).to(self.device)

    def forward(self, x):
        return self.conv(x)

class TransCov(nn.Module):
    def __init__(self, in_channels, device):
        super(TransCov, self).__init__()
        self.device = device
        self.trans_conv = nn.ConvTranspose2d(in_channels*2, in_channels, 2, 2, device=device)

    def forward(self, x):
        return self.trans_conv(x)

class Unet(nn.Module):
    def __init__(self, out_channels, device):
        super(Unet, self).__init__()
        self.device = device
        self.list = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2, 2)
        self.f_out = nn.Conv2d(self.list[0], out_channels, 1)

    def forward(self, x):
        in_channels = x.shape[1]
        activation = []

        '''Conv and max-pool layers'''
        for channels in self.list:
            x = ConvCat(in_channels, channels, self.device)(x)
            # store activation
            activation.append(x)
            # pooling layer
            x = self.pool(x)
            in_channels = x.shape[1]

        x = ConvCat(self.list[-1], self.list[-1]*2, self.device)(x)

        # reversing the order of activation layers
        activation = activation[::-1]

        '''Transpose Conv and conv layers'''
        for idx, channels in enumerate(reversed(self.list)):
            x = TransCov(channels, self.device)(x)
            active = activation[idx]

            # here we should reshape the activation output (according to the paper)
            # but we are reshaping the trans conv layer
            if active.shape != x.shape:
                x = TF.resize(x, size=active.shape[2:])

            x = torch.cat((active, x), dim=1)

            # again conv-cat series
            x = ConvCat(channels*2, channels, self.device)(x)

        # the out layer
        x = self.f_out(x)
        return x

if __name__ == '__main__':
    a = torch.zeros((2, 1, 160, 160)).to('cuda')

    model = Unet(4, 'cuda').to(device='cuda')
    print(model(a).shape)

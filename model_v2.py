import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# In original paper the output shape is not similar to the input shape
# Here we are going to make both similar that's why I decided to put padding in Convolutional layer

class ConvCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvCat, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.list = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.f_out = nn.Conv2d(self.list[0], out_channels, 1)

        for feature in self.list:
            self.down.append(ConvCat(in_channels, feature))
            in_channels = feature

        for feature in reversed(self.list):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, 2, 2))
            self.ups.append(ConvCat(feature*2, feature))

        self.bottom_layer = ConvCat(self.list[-1], self.list[-1]*2)

    def forward(self, x):
        # for the skip connections
        activation = []

        '''Conv and max-pool layers'''
        for down in self.down:
            x = down(x)
            activation.append(x)
            x = self.pool(x)

        x = self.bottom_layer(x)

        # reversing the order of activation layers for adding
        # these as a skip connection
        activation = activation[::-1]

        '''Transpose Conv and conv layers'''
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            active = activation[idx//2]

            # here we should reshape the activation output (according to the paper)
            # but we are reshaping the trans conv layer
            if active.shape != x.shape:
                x = TF.resize(x, size=active.shape[2:])

            # adding the skip connection
            x = torch.cat((active, x), dim=1)

            # again conv-cat series
            x = self.ups[idx+1](x)

        # the out layer
        x = self.f_out(x)
        return x

if __name__ == '__main__':
    a = torch.zeros((2, 1, 160, 160)).to('cuda')

    model = Unet(1, 4).to(device='cuda')
    print(model(a).shape)

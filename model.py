"""
Implementation of VGGNet, from paper
"""
import torch
import torch.nn as nn


# example configurations for various types of VGGNets
# idea borrowed from : https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py#L62
VGG_CONFS = {
    'vgg11': (64, 'max', 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'),
    'vgg13': (64, 64, 'max', 128, 128, 'max', 256, 256, 'max', 512, 512, 'max', 512, 512, 'max'),
    'vgg16': (64, 64, 'max', 128, 128, 'max', 256, 256, 256, 'max', 512, 512, 512, 'max', 512, 512, 512, 'max'),
    'vgg19': (64, 64, 'max', 128, 128, 'max', 256, 256, 256, 256, 'max', 512, 512, 512, 512, 'max', 512, 512, 512, 512, 'max'),
}


class VGGNet(nn.Module):
    """
    Neural network model consisting of layers propsed by VGGNet paper.
    """
    def __init__(self, config, dim=224, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Example layers for VGG-16, following Table-1 of https://arxiv.org/pdf/1409.1556.pdf,
        ("Simonyan et al. - Very Deep Convolutional Networks for Large Scale Image Recognition"):

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (b x 64 x 224 x 224)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),  # (b x 64 x 224 x 224)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b x 64 x 112 x 112)

            nn.Conv2d(64, 128, 3, 1, 1),  # (b x 128 x 112 x 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),  # (b x 128 x 112 x 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (b x 64 x 56 x 56)

            nn.Conv2d(128, 256, 3, 1, 1),  # (b x 256 x 56 x 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (b x 256 x 56 x 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # (b x 256 x 56 x 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (b x 256 x 28 x 28)

            nn.Conv2d(256, 512, 3, 1, 1),  # (b x 512 x 28 x 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (b x 512 x 28 x 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (b x 512 x 28 x 28)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (b x 512 x 14 x 14)

            nn.Conv2d(512, 512, 3, 1, 1),  # (b x 512 x 14 x 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (b x 512 x 14 x 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),  # (b x 512 x 14 x 14)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (b x 512 x 7 x 7)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=(512 * 7 * 7), out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        self.net, dim_shrink_rate, out_channels = self.create_layers(config)
        dim_feat = dim // dim_shrink_rate
        if dim_feat == 0:
            print(self.net)
            raise ValueError('Image dimension too small for this network: '
                             'Should have dimensions larger than {}'.format(dim_shrink_rate))

        self.linear_input_size = dim_feat * dim_feat * out_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.linear_input_size, out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        self.apply(self.init_weights)  # initialize weights

    def create_layers(self, config):
        """Borrowed this idea from : https://github.com/chengyangfu/pytorch-vgg-cifar10"""
        channels = 3
        dim_shrink_rate = 1
        layers = []
        for layer in config:
            if layer == 'max':  # create maxpool layer
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                dim_shrink_rate *= 2
            elif isinstance(layer, int):  # create convolutional layer + activation
                layers.extend([nn.Conv2d(in_channels=channels, out_channels=layer, kernel_size=3, padding=1),
                               nn.ReLU(inplace=True)])
                channels = layer
            else:
                raise TypeError(
                    'layer in config should be either "max" or an int denoting the channel size. '
                    'Got type {} instead.'.format(type(layer)))
        return nn.Sequential(*layers), dim_shrink_rate, channels

    def forward(self, x):
        """
        Pass the input through the net.

        Args:
            x (Tensor): input tensor

        Returns:
            output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1, self.linear_input_size)  # reduce the dimensions for linear layer input
        return self.classifier(x)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = VGGNet(VGG_CONFS['vgg16'], dim=32, num_classes=10)
    sample_data = torch.randn((10, 3, 32, 32))
    out = net(sample_data)
    print(out)
    print(out.size())

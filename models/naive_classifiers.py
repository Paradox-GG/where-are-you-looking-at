from torch import nn
import torch


def conv_layer(ch_in, ch_out, ks=3, st=1, dl=1, pd=0, activate=nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=ks, stride=st, dilation=dl, padding=pd),
        nn.BatchNorm2d(ch_out),
        activate
    )


class Naive_CNN(nn.Module):
    def __init__(self):
        super(Naive_CNN, self).__init__()
        self.l1 = conv_layer(ch_in=3, ch_out=16, st=2, pd=1)
        self.l2 = conv_layer(ch_in=16, ch_out=32, st=2)
        self.l3 = conv_layer(ch_in=32, ch_out=64, st=2)
        self.l4 = conv_layer(ch_in=64, ch_out=64, ks=2)

        self.branch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.branch(x)
        x = x.squeeze()
        return x


class Naive_RNN(nn.Module):
    def __init__(self, in_feature=32, hidden_feature=100, num_class=4, num_layers=2):
        super(Naive_RNN, self).__init__()
        self.rnn = nn.LSTM(in_feature, hidden_feature, num_layers)
        self.classifier = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        x = x[:, 0, :, :]
        x = x.permute(1, 0, 2)
        out, _ = self.rnn(x)
        out = out[-1, :, :]
        out = self.classifier(out)
        return out

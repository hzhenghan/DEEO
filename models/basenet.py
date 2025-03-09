import torch
import torch.nn.functional as F
import torch.nn as nn



class ResClassifier_MME(nn.Module):
    def __init__(self, num_classes=12, input_size=2048, temp=0.05236763854, norm=True):
        super(ResClassifier_MME, self).__init__()
        if norm:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        else:
            self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.norm = norm
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        if self.norm:
            x = F.normalize(x)
            x = self.fc(x)/self.tmp
        else:
            x = self.fc(x)
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        S = 16
        N = 4

        self.f_conv1 = nn.Conv1d(1, 16, 4, 2, 1, bias=False)
        self.f_relu1 = nn.LeakyReLU(True)
        self.f_conv2 = nn.Conv1d(16, 32, 4, 2, 1, bias=False)
        self.f_bn2 = nn.BatchNorm1d(32)
        self.f_relu2 = nn.LeakyReLU(True)
        self.f_pool1 = nn.MaxPool1d(2)
        self.f_relu3 = nn.LeakyReLU(True)
        self.f_conv3 = nn.Conv1d(32, 32, 4, 2, 1, bias=False)
        self.f_bn3 = nn.BatchNorm1d(32)
        self.f_relu4 = nn.LeakyReLU(True)
        self.f_conv4 = nn.Conv1d(32, 32, 4, 2, 1, bias=False)
        self.f_bn4 = nn.BatchNorm1d(32)
        self.f_relu5 = nn.LeakyReLU(True)
        self.f_pool2 = nn.MaxPool1d(2)
        self.f_relu6 = nn.LeakyReLU(True)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32, 64)

    def forward(self, x):
        # print('shape of input is {}'.format(x.shape))
        # x = self.Vit(x)
        x = x.float()
        x = self.f_conv1(x)
        x = self.f_relu1(x)
        x = self.f_conv2(x)
        x = self.f_bn2(x)
        x = self.f_relu2(x)
        x = self.f_pool1(x)
        x = self.f_relu3(x)
        x = self.f_conv3(x)
        x = self.f_bn3(x)
        x = self.f_relu4(x)
        x = self.f_conv4(x)
        x = self.f_bn4(x)
        x = self.f_relu6(x)
        x = self.flatten(x)

        return x
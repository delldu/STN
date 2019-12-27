# encoding: utf-8

# import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
# from torch.autograd import Variable
from tps_grid_gen import TPSGridGen
import pdb


class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)
        # pdb.set_trace()
        # (Pdb) a
        # self = CNN(
        #   (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
        #   (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
        #   (conv2_drop): Dropout2d(p=0.5, inplace=False)
        #   (fc1): Linear(in_features=320, out_features=50, bias=True)
        #   (fc2): Linear(in_features=50, out_features=32, bias=True)
        # )
        # num_output = 32

    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) pp x.size()
        # torch.Size([64, 1, 28, 28])
        # (Pdb) pp self.conv1(x).size()
        # torch.Size([64, 10, 24, 24])

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # pdb.set_trace()
        # torch.Size([64, 32])
        return x


class ClsNet(nn.Module):
    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))


class BoundedGridLocNet(nn.Module):
    def __init__(self, grid_height, grid_width, target_control_points):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = torch.from_numpy(np.arctanh(target_control_points.numpy()))
        bias = bias.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)


class UnBoundedGridLocNet(nn.Module):
    def __init__(self, grid_height, grid_width, target_control_points):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_height * grid_width * 2)

        bias = target_control_points.view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)


class STNClsNet(nn.Module):
    def __init__(self, args):
        super(STNClsNet, self).__init__()
        self.args = args

        r1 = args.span_range_height  # 0.9
        r2 = args.span_range_width  # 0.9
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
        target_control_points = torch.Tensor(
            list(
                itertools.product(
                    np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (args.grid_height - 1)),
                    np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (args.grid_width - 1)),
                )))
        # pdb.set_trace()
        # # pp 2.0 * r1 / (args.grid_height - 1) -- 0.6
        # array([-0.9, -0.3,  0.3,  0.9])
        # pp target_control_points
        # tensor([[-0.9000, -0.9000],
        #         [-0.9000, -0.3000],
        #         [-0.9000,  0.3000],
        #         [-0.9000,  0.9000],
        #         [-0.3000, -0.9000],
        #         [-0.3000, -0.3000],
        #         [-0.3000,  0.3000],
        #         [-0.3000,  0.9000],
        #         [ 0.3000, -0.9000],
        #         [ 0.3000, -0.3000],
        #         [ 0.3000,  0.3000],
        #         [ 0.3000,  0.9000],
        #         [ 0.9000, -0.9000],
        #         [ 0.9000, -0.3000],
        #         [ 0.9000,  0.3000],
        #         [ 0.9000,  0.9000]])

        Y, X = target_control_points.split(1, dim=1)
        target_control_points = torch.cat([X, Y], dim=1)

        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[args.model]
        self.loc_net = GridLocNet(args.grid_height, args.grid_width,
                                  target_control_points)

        self.tps = TPSGridGen(args.image_height, args.image_width,
                              target_control_points)

        self.cls_net = ClsNet()
        # pdb.set_trace()
        # (Pdb) a
        # self = STNClsNet(
        #   (loc_net): UnBoundedGridLocNet(
        #     (cnn): CNN(
        #       (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
        #       (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
        #       (conv2_drop): Dropout2d(p=0.5, inplace=False)
        #       (fc1): Linear(in_features=320, out_features=50, bias=True)
        #       (fc2): Linear(in_features=50, out_features=32, bias=True)
        #     )
        #   )
        #   (tps): TPSGridGen()
        #   (cls_net): ClsNet(
        #     (cnn): CNN(
        #       (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
        #       (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
        #       (conv2_drop): Dropout2d(p=0.5, inplace=False)
        #       (fc1): Linear(in_features=320, out_features=50, bias=True)
        #       (fc2): Linear(in_features=50, out_features=10, bias=True)
        #     )
        #   )
        # )
        # args = Namespace(angle=90, batch_size=64, cuda=True, epochs=10,
        # grid_height=4, grid_size=4, grid_width=4, image_height=28,
        # image_width=28, log_interval=10, lr=0.01, model='unbounded_stn',
        # momentum=0.5, no_cuda=False, save_interval=100, seed=1, span_range=0.9,
        # span_range_height=0.9, span_range_width=0.9, test_batch_size=1000)

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, self.args.image_height,
                                      self.args.image_width, 2)
        transformed_x = grid_sample(x, grid)
        logit = self.cls_net(transformed_x)
        # pdb.set_trace()
        # (Pdb) pp source_control_points.size()
        # torch.Size([64, 16, 2])
        # (Pdb) source_coordinate.size()
        # torch.Size([64, 784, 2])
        # (Pdb) grid.size()
        # torch.Size([64, 28, 28, 2])
        # (Pdb) transformed_x.size()
        # torch.Size([64, 1, 28, 28])
        # (Pdb) logit.size()
        # torch.Size([64, 10])
        return logit


def get_model(args):
    if args.model == 'no_stn':
        print('create model without STN')
        model = ClsNet()
    else:
        print('create model with STN')
        model = STNClsNet(args)
    return model

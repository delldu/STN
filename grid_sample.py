# encoding: utf-8

import torch.nn.functional as F
# from torch.autograd import Variable
import pdb


def grid_sample(input, grid, canvas=None):
    # pdb.set_trace()
    # torch.Size([64, 1, 28, 28])
    # (Pdb) pp grid.size()
    # torch.Size([64, 28, 28, 2])

    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import darts_ops_x as ops
import torch.nn.functional as F
import numpy as np
import itertools


# training cell
class Training_Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Training_Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)
        genotype_reduce = list(itertools.chain.from_iterable(genotype.reduce))
        genotype_normal = list(itertools.chain.from_iterable(genotype.normal))
        if reduction_prev:
            self.preprocess0 = ops.FactorizedReduce(C_prev_prev, C, skip=False)
        else:
            self.preprocess0 = ops.ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ops.ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype_reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype_normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = ops.OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, ops.Identity):
                    h1 = ops.drop_path_(h1, drop_prob, self.training)
                if not isinstance(op2, ops.Identity):
                    h2 = ops.drop_path_(h2, drop_prob, self.training)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
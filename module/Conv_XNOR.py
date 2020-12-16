import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.conv import _ConvNd
from module.utils import _pair

class Sigh_XNOR(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        ctx.alpha = torch.mean(torch.abs(input))
        #print(ctx.alpha,'xishu')
        return ctx.alpha * torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input_part1 = grad_output / torch.numel(grad_output)
        grad_input_part2 = grad_output.clone()
        grad_input_part2[ctx.input > 1] = 0
        grad_input_part2[ctx.input < -1] = 0
        grad_input = grad_input_part1 + ctx.alpha * grad_input_part2
        return grad_input
        

class Conv_XNOR(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                    padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv_XNOR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.Sigh_XNOR = Sigh_XNOR.apply

    def forward(self, x):
        binarized_weight = self.Sigh_XNOR(self.weight)
        return F.conv2d(x, binarized_weight, self.bias, self.stride,
                self.padding, self.dilation, self.groups)
        
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

def main():
    conv1 = Conv_XNOR(3, 16, 3, padding=1).cuda()
    input = torch.randn(4, 3, 32, 32, requires_grad=True).cuda()
    output = conv1(input)
    output.backward(torch.ones_like(output).cuda())
    print(output)
    print(conv1.weight.grad)

if __name__ == '__main__':
    main()

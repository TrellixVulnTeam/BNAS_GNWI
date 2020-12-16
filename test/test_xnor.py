import os
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
import sys
# setting the global param
root_path = './results'
data_path = {'imagenet': '/data/ImageNet',
             'cifar10': '/data/CIFAR',
             }
data_size = {'imagenet': '/data/ImageNet',
             'cifar10': '/data/CIFAR',
             }
parser = argparse.ArgumentParser("train_parser")
# data argument
parser.add_argument('--data', type=str, default='cifar10', choices=['imagenet', 'cifar10'], help='dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--cutout', type=int, default=0, help='cutout')
parser.add_argument('--workers', type=int, default=4, help='worker to load the image')
# model
parser.add_argument('--init_channels', type=int, default=44, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--drop_out', type=float, default=0, help='drop out probability')
parser.add_argument('--model_name', type=str, default='BNAS_XNOR_1', help='model name ')
parser.add_argument('--resume', '-r', type=str, default='./weights/XNOR_larger.pth.tar', help='resume from checkpoint')
# training
parser.add_argument('--print_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', default="3", help='gpu device id')
parser.add_argument('--epochs', type=int, default=1, help='num of training epochs')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=[None, 'bn', 'bn#bias'])
parser.add_argument('--no_nesterov', action='store_true')  # opt_param

parser.add_argument('--save', type=str, default='test', help='save dir name')
parser.add_argument('--manual_seed', default=0, type=int)
args = parser.parse_args()


save_dir_str = args.data + '_' \
               + time.asctime(time.localtime()).replace(' ', '_') + '_' + args.model_name

out_path = os.path.join(root_path, args.save, save_dir_str)
# set GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.gpu = [int(i) for i in args.gpu.split(',')]
# the environ should before import torch!
import torch
import torch.nn as nn
sys.path.append('.')
from models.darts_cnn_x import NetworkCIFAR
from utils import utils
from data_loader import get_data

from utils.genotypes_x import genotype_array
# set device
torch.cuda.set_device(args.gpu[0])
device = torch.device("cuda")
os.makedirs(out_path)
logger = utils.get_logger(os.path.join(out_path, "logger.log"))
# set logger
logger.info("Logger is set - training start")
utils.print_params(vars(args), logger.info)
# set seed
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
torch.backends.cudnn.benchmark = True
best_top1 = 0


def main():
    pin_memory = False if args.data == 'cifar10' else True
    [input_size, input_channels, n_classes, train_data, val_data] = \
        get_data.get_data(args.data, data_path[args.data], args.cutout, True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=args.batch_size,
                                                sampler=None,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=pin_memory)
    genotype = genotype_array[args.model_name]

    model = NetworkCIFAR(args.init_channels, n_classes, args.layers, args.auxiliary, genotype,
                                    drop_out=args.drop_out)
    model.set_bn_param(args.bn_momentum, args.bn_eps)
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} M".format(mb_params))

    
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    start_epoch = 0
    global best_top1
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        # assert os.path.isdir(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['model'])
        # best_top1 = checkpoint['best_top1']

        # logger.info("Best Prec@1 = {:.4%}".format(best_top1))

    test_criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, args.epochs):

        # validation
        top1 = validate(val_loader, model, test_criterion, epoch)
        logger.info("Current Prec@1 = {:.4%}".format(top1))

        print("")



def validate(valid_queue, model, criterion, epoch):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.module.eval()
    len_val_quene = len(valid_queue)

    with torch.no_grad():
        for step, data in enumerate(valid_queue):
            input = data[0].to(device)
            target = data[1].to(device)

            result = model(input)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result
            loss = criterion(logits, target)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            if step % args.print_freq == 0 or step == len_val_quene - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, args.epochs, step, len_val_quene - 1, losses=objs,
                        top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
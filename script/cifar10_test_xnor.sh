#!/usr/bin/env bash
python ./test/test_xnor.py \
        --data cifar10 --batch_size 512 --workers 16 --model_name "BNAS_XNOR_1" --auxiliary --init_channels 36 --resume './weights/XNOR.pth.tar'
#!/usr/bin/env bash

cd ..




python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=192 \
--Group_number=16 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=192 \
--Group_number=8 \
--group_number=2 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=192 \
--Group_number=8 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=224 \
--Group_number=8 \
--group_number=2 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=224 \
--Group_number=8 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=224 \
--Group_number=16 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=5 \
--kernal_number=224 \
--Group_number=8 \
--group_number=2 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=5 \
--kernal_number=224 \
--Group_number=8 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'

python train_cifar.py \
--arch=SRGC_V4 \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=5 \
--kernal_number=224 \
--Group_number=16 \
--group_number=1 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 \
--ckpt_path='/home/amax/LR/ResNet/LSTM'





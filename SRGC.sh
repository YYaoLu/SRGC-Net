#!/usr/bin/env bash


python train_cifar.py \
--arch=SRGC \
--dataset=cifar10 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--kernal_number=160 \
--Group_number=4 \
--group_number=4 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 \
--width_factor=2 



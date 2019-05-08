#!/usr/bin/env bash
cd ..

python CIFAR_train.py \
--arch=SRGC_Nets \
--dataset=cifar10 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--filter=96 \
--Group_number=4 \
--group_number=4 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 



python CIFAR_train.py \
--arch=SRGC_Nets \
--dataset=cifar100 \
--epochs=200 \
--start_epoch=0 \
--batch_size=128 \
--block_number=4 \
--filter=96 \
--Group_number=4 \
--group_number=4 \
--lr=0.1 \
--lr_schedule=0 \
--momentum=0.9 \
--weight_decay=0.0002 





# SRGC-Net

# Requirements
This code was developed and tested with Python2.7, Pytorch 0.3 and CUDA 8.0 on Ubuntu 14.04.

# Running the Demo
You are able to run the provided demo code.  
```
sh SRGC.sh
```
# How to train
You should be able to train the model by running the following command 
```
python train_cifar.py --arch=net_style(SRGC) --dataset==/path/to/cifar/or/imagenet --ckpt_path=/path/to/checkpoint/ --resume --epochs=200 --start_epoch=0 --batch_size=128 --block_number=4 --kernel_number=180 --Group_number=4 --group_number=4 --lr=0.1 --lr_schedule=0 --momentum=0.9 --weight_decay=0.0002 --width_factor=2 
```

# FDA
code for "FDA: Attention Module for Multi-scale Feature Descriptions through Local and Global Fusions"

## RUN
Single GPU Training
```
python train.py -c schedules/cifar_bs128.yaml -p /data/datasets -s cifar100 -a FDA -n resnet56 --device 0
```

DP Training
```
python train.py -c schedules/cifar_bs128.yaml -p /data/datasets -s cifar100 -a FDA -n resnet56
```

DDP Training
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 8888 train.py -c schedules/imagenet_bs32x8.yaml -p /data/datasets -s imagenet -a FDA -n ResNet50
```

## Our Environment
python 3.8.16
pytorch 1.12.1
CUDA 11.1
Ubuntu 22.04

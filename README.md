# FDA
code for "Feature Description Attention: Channel-independent local–global fusion for multi-scale feature representation"

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
- python 3.8.16
- pytorch 1.12.1
- CUDA 11.1
- Ubuntu 22.04


## Citation
If you use FDA in your work, please use the following BibTeX entries:
```
@article{zhu2025feature,
         title = {Feature Description Attention: Channel-independent local–global fusion for multi-scale feature representation},
         author = {Yuanyang Zhu and Guangjie Han and Hongbo Zhu and Fan Zhang}
         journal = {Engineering Applications of Artificial Intelligence},
         volume = {161},
         pages = {112139},
         year = {2025},
         issn = {0952-1976},
         doi = {https://doi.org/10.1016/j.engappai.2025.112139}
}
```

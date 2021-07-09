# Novel Slot Detection: A Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System
This repository is the official implementation of [Novel Slot Detection: A Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System.](https://arxiv.org/abs/2105.14313v1)(**ACL2021**) by [Yanan Wu](https://aclanthology.org/people/y/yanan-wu/), [Zhiyuan Zeng](), [Keqing He](https://aclanthology.org/people/z/zhiyuan-zeng/), [Hong Xu](https://www.aclweb.org/anthology/people/h/hong-xu/),  [Yuanmeng Yan](https://www.aclweb.org/anthology/people/y/yuanmeng-yan/), [Huixing Jiang](https://aclanthology.org/people/h/huixing-jiang/), [Weiran Xu](https://www.aclweb.org/anthology/people/w/weiran-xu/). 

## Introduction
The Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System.

An example of Novel Slot Detection in thetask-oriented dialogue system:

![](https://github.com/ChestnutWYN/ACL2021-Novel-Slot-Detection/blob/main/fig/intro.png)

The architecture of the proposed model:

![](https://github.com/ChestnutWYN/ACL2021-Novel-Slot-Detection/blob/main/fig/model.jpg)


## Dependencies

We use anaconda to create python environment:
```
conda create --name python=3.6
```
Install all required libraries:
```
pip install -r requirements.txt
```

## How to run
#### 1. Train (only):
```
python --mode train --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 
```
#### 2. Predict (only):
```
python --mode test --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 

```
#### 1. Train and predict (Both):
```
python --mode both --dataset SnipsNSD5% --threshold 8.0 --output_dir ./output --batch_size 256 --cuda 1 

```
## Parameters
- `mode`, optional, Specify running mode, only `train`,only`test` or `both`.
- `dataset`, required, The dataset to use, `SnipsNSD5%` or `SnipsNSD15%` or `SnipsNSD30%`.
- `threshold`, required, The specified threshold value.
- `output_dir`, default="./output"
- `batch_size`, default=256
- `cuda`, default=1
## Result

## Citation
```
@article{Wu2021NovelSD,
  title={Novel Slot Detection: A Benchmark for Discovering Unknown Slot Types in the Task-Oriented Dialogue System},
  author={Yanan Wu and Zhiyuan Zeng and Keqing He and Hong Xu and Yuanmeng Yan and Huixing Jiang and Weiran Xu},
  journal={ArXiv},
  year={2021},
  volume={abs/2105.14313}
}
```
## Issue
Q：**There are two training objectives mentioned in Section 4.1: multiple classifier and binary classifier. But if we use binary classifier, how can we get the ind category? And how to get the results of MSP + binary and GDA + binary?**

A：As we mention in Section4.1—— "In the test stage, for in-domain prediction, we both use the multiple classifier. While, for novel slot detection, we use the multiple classifier or the binary classifier, or both of them". It means binary classifier won't be used for gaining the fine in-domain labels, but for detecting whether a token is a novel slot, and if yes, we will override the fine in-domain labels gained by multiple classifier.


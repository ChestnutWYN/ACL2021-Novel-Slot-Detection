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

1. IND and NSD results with different proportions (5%, 15% and 30%) of classes are treated as unknown
slots on Snips-NSD. * indicates the significant improvement over all baselines (p < 0.05).

<table>
      <tr  align="center">
        <td colspan="3"><b></b></td>
        <td colspan="3"><b>5%</b></td>
        <td colspan="3"><b>15%</b></td>
        <td colspan="3"><b>30%</b></td>
    </tr>
      <tr  align="center">
           <td colspan="3"><b>Models</b></td>
            <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
            <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
        <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
        </tr>
      <tr  align="center">
            <td><b>detection method</b></td>
            <td><b>objective</b></td>
            <td><b>distance strategy</b></td>
            <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
             <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>binary</td>
            <td>-</td>
            <td>87.21 </td>
            <td>12.34 </td>
            <td>25.16 </td>
            <td>71.44 </td>
            <td>12.31 </td>
            <td>39.50 </td>
            <td>58.88 </td>
            <td>8.73 </td>
            <td>40.38 </td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>multiple</td>
            <td>-</td>
            <td>88.05 </td>
            <td>14.04 </td>
            <td>30.50 </td>
            <td>79.71 </td>
            <td>20.97 </td>
            <td>40.02 </td>
            <td>78.52 </td>
            <td>25.26 </td>
            <td>46.91 </td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>binary+multiple</td>
            <td>-</td>
            <td>89.59 </td>
            <td>23.58 </td>
            <td>37.55 </td>
            <td>83.72 </td>
            <td>24.70 </td>
            <td>45.32 </td>
            <td>79.08 </td>
            <td>30.66 </td>
            <td>52.10 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>binary</td>
            <td>difference</td>
            <td>87.95 </td>
            <td>23.83 </td>
            <td>35.83 </td>
            <td>83.65 </td>
            <td>22.06 </td>
            <td>43.99 </td>
            <td>78.72 </td>
            <td>32.50 </td>
            <td>44.13 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>binary</td>
            <td>minumum</td>
            <td>61.29 </td>
            <td>10.36 </td>
            <td>17.08 </td>
            <td>49.11 </td>
            <td>16.91 </td>
            <td>31.10 </td>
            <td>48.07 </td>
            <td>15.56 </td>
            <td>33.78 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>multiple</td>
            <td>difference</td>
            <td>93.14 </td>
            <td>29.73 </td>
            <td>45.99 </td>
            <td>90.07 </td>
            <td>31.96 </td>
            <td>53.02 </td>
            <td>85.56 </td>
            <td>36.16 </td>
            <td>54.55 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>multiple</td>
            <td>minumum</td>
            <td>93.10 </td>
            <td>31.67*</td>
            <td>46.97*</td>
            <td>90.18 </td>
            <td>32.19 </td>
            <td>53.75*</td>
            <td>86.26*</td>
            <td>38.64*</td>
            <td>55.24*</td>
        </tr>
</table>

2.  IND and NSD results with different proportions (5%, 15% and 30%) of classes are treated as unknown
slots on ATIS-NSD. * indicates the significant improvement over all baselines (p < 0.05).

<table>
      <tr  align="center">
        <td colspan="3"><b></b></td>
        <td colspan="3"><b>5%</b></td>
        <td colspan="3"><b>15%</b></td>
        <td colspan="3"><b>30%</b></td>
    </tr>
      <tr  align="center">
           <td colspan="3"><b>Models</b></td>
            <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
            <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
        <td><b>IND</b></td>
                <td colspan="2"><b>NSD</b></td>
        </tr>
      <tr  align="center">
            <td><b>detection method</b></td>
            <td><b>objective</b></td>
            <td><b>distance strategy</b></td>
            <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
             <td><b>Span F1</b></td>
            <td><b>Span F1</b></td>
            <td><b>Token F1</b></td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>binary</td>
            <td>-</td>
            <td>92.04 </td>
            <td>19.73 </td>
            <td>29.63 </td>
            <td>91.74 </td>
            <td>23.40 </td>
            <td>33.89 </td>
            <td>80.49 </td>
            <td>21.88 </td>
            <td>39.17 </td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>multiple</td>
            <td>-</td>
            <td>94.33 </td>
            <td>27.15 </td>
            <td>31.16 </td>
            <td>92.54 </td>
            <td>39.88 </td>
            <td>42.29 </td>
            <td>87.63 </td>
            <td>40.42 </td>
            <td>47.64 </td>
        </tr>
      <tr  align="center">
            <td>MSP</td>
            <td>binary+multiple</td>
            <td>-</td>
            <td>94.41 </td>
            <td>32.49 </td>
            <td>43.48 </td>
            <td>93.29 </td>
            <td>41.23 </td>
            <td>43.13 </td>
            <td>90.14 </td>
            <td>41.76 </td>
            <td>51.87 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>binary</td>
            <td>difference</td>
            <td>93.69 </td>
            <td>27.02 </td>
            <td>34.21 </td>
            <td>92.13 </td>
            <td>30.51 </td>
            <td>36.30 </td>
            <td>88.73 </td>
            <td>30.91 </td>
            <td>45.64 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>binary</td>
            <td>minumum</td>
            <td>93.57 </td>
            <td>15.90 </td>
            <td>20.96 </td>
            <td>90.98 </td>
            <td>24.53 </td>
            <td>27.26 </td>
            <td>88.21 </td>
            <td>26.40 </td>
            <td>39.83 </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>multiple</td>
            <td>difference</td>
            <td>95.20 </td>
            <td>47.78* </td>
            <td>51.54* </td>
            <td>93.92 </td>
            <td>50.92* </td>
            <td>52.24* </td>
            <td>92.02 </td>
            <td>51.26* </td>
            <td>56.59* </td>
        </tr>
      <tr  align="center">
            <td>GDA</td>
            <td>multiple</td>
            <td>minumum</td>
            <td>95.31* </td>
            <td>41.74 </td>
            <td>45.91 </td>
            <td>93.88 </td>
            <td>43.78 </td>
            <td>46.18 </td>
            <td>91.67 </td>
            <td>45.44 </td>
            <td>52.37 </td>
        </tr>
</table>

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


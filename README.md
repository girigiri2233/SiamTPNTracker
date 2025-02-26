# SiamTPN

## Introduction

This is the official implementation of the SiamTPN (WACV2022). The tracker intergrates pyramid feature network and transformer into Siamese network, achieving state-of-the-art performance (better than DiMP) while runing 30 FPS on a single CPU.  The tracker optimized with ONXX and openvino could run at 45 FPS on cpu end, leading promising performance when deploying on drones for tracking.

![AO_Speed_GOT10K](imgs/got10_ao_speed.png)

[\[Paper\]](https://arxiv.org/abs/2110.08822)  [\[Raw Results\]](https://drive.google.com/drive/folders/1YUzqgifqhXVK_PrQNg5w467EGlX9yG-M?usp=sharing)   [\[Drone Tracking Videos\]](https://drive.google.com/drive/folders/1joOkom2sDZ-Ke2eyYIQX5redjpku38Lk?usp=sharing)  [\[Models\]](https://drive.google.com/drive/folders/1Aryamx7-UP9G3R9_Zn7IJ95Iyl3soVQx?usp=sharing)

## Training

### prepare data
change the path in `lib/train/admin/local.py ` to your data location

```
# Distributed training withh 4 nodes 
python -m torch.distributed.launch --nproc_per_node 4 tools/run_training.py --config shufflenet_l345_192
```

```
# single gpu training for test purpose
python tools/run_training.py --config shufflenet_l345_192
```


## Test and evaluate SiamTPN

### prepare data
change the path in `lib/test/evaluation/local.py ` to your data location

### running on cpu 

```
# Download the pretrain model and put it under ./results/checkpoints/train/SiamTPN/ folder

python tools/test.py siamtpn shufflenet_l345_192 --dataset_name got10k_val --debug 1 --cpu 1 --epoch 100 --sequence GOT-10k_Val_000001
```

### running on cpu with onnx optimized

The debug mode will show tracking results, more details refer to tools/test.py

Currently, onnx only support cpu version

First, you need to install onxx and onxxruningtime:

```
pip install onxx
# for onxx runining time, download the openvino version from release [page](https://github.com/intel/onnxruntime/releases/tag/v3.1) and install with
pip install onnxruntime_openvino-1.9.0-cp37-cp37m-linux_x86_64.whl

# please refer the [page](https://github.com/intel/onnxruntime/releases/tag/v3.1) for openvino installation details.
```

```
# Download the converted onnx model and put it under ./results/onnx/ folder
# or conver your own model with 
python tools/onnx_search.py
python tools/onnx_template.py

python tools/test.py siamtpn_onnx shufflenet_l345_192 --dataset_name got10k_val --debug 1 --cpu 1 --epoch 100 --sequence GOT-10k_Val_000001
```


## Citation
If you find this repo useful, please cite with
```
@article{xing2021siamese,
  title={Siamese Transformer Pyramid Networks for Real-Time UAV Tracking},
  author={Xing, Daitao and Evangeliou, Nikolaos and Tsoukalas, Athanasios and Tzes, Anthony},
  journal={arXiv preprint arXiv:2110.08822},
  year={2021}
}
```


## Acknowledge
Our code is implemented based on the following libraries:
* [PyTracking](https://github.com/visionml/pytracking)
* [Stark](https://github.com/researchmm/Stark)
* [Timm](https://github.com/rwightman/pytorch-image-models)



好的，以下是将README.md中的必要部分翻译成中文的内容：

### SiamTPN

#### 简介

这是SiamTPN（WACV2022）的官方实现。该跟踪器将金字塔特征网络和Transformer集成到Siamese网络中，在单个CPU上实现了30 FPS的速度，性能优于DiMP。经过ONNX和OpenVINO优化后，该跟踪器在CPU端可以达到45 FPS，展现出在无人机部署时的卓越性能。

![AO_Speed_GOT10K](imgs/got10_ao_speed.png)

[【论文】](https://arxiv.org/abs/2110.08822)  [【原始结果】](https://drive.google.com/drive/folders/1YUzqgifqhXVK_PrQNg5w467EGlX9yG-M?usp=sharing)   [【无人机跟踪视频】](https://drive.google.com/drive/folders/1joOkom2sDZ-Ke2eyYIQX5redjpku38Lk?usp=sharing)  [【模型】](https://drive.google.com/drive/folders/1Aryamx7-UP9G3R9_Zn7IJ95Iyl3soVQx?usp=sharing)

#### 训练

##### 准备数据
更改 `lib/train/admin/local.py` 中的数据路径为你自己的数据位置。

```bash
# 分布式训练（4个节点）
python -m torch.distributed.launch --nproc_per_node 4 tools/run_training.py --config shufflenet_l345_192
```

```bash
# 单GPU训练（测试用途）
python tools/run_training.py --config shufflenet_l345_192
```

#### 测试和评估SiamTPN

##### 准备数据
更改 `lib/test/evaluation/local.py` 中的数据路径为你自己的数据位置。

##### 在CPU上运行

```bash
# 下载预训练模型并将其放在 ./results/checkpoints/train/SiamTPN/ 文件夹下

python tools/test.py siamtpn shufflenet_l345_192 --dataset_name got10k_val --debug 1 --cpu 1 --epoch 100 --sequence GOT-10k_Val_000001
```

##### 使用ONNX优化在CPU上运行

调试模式会显示跟踪结果，更多详情请参阅 `tools/test.py`

目前，ONNX仅支持CPU版本。

首先，你需要安装ONNX和ONNXRuntime：

```bash
pip install onnx
# 对于ONNXRuntime，请从发布页面下载OpenVINO版本并安装：
pip install onnxruntime_openvino-1.9.0-cp37-cp37m-linux_x86_64.whl

# 请参考[页面](https://github.com/intel/onnxruntime/releases/tag/v3.1)获取OpenVINO安装详情。
```

```bash
# 下载转换后的ONNX模型并将其放在 ./results/onnx/ 文件夹下
# 或者使用以下命令转换你自己的模型：
python tools/onnx_search.py
python tools/onnx_template.py

python tools/test.py siamtpn_onnx shufflenet_l345_192 --dataset_name got10k_val --debug 1 --cpu 1 --epoch 100 --sequence GOT-10k_Val_000001
```

#### 引用
如果你觉得这个仓库对你有帮助，请引用如下：

```bibtex
@article{xing2021siamese,
  title={Siamese Transformer Pyramid Networks for Real-Time UAV Tracking},
  author={Xing, Daitao and Evangeliou, Nikolaos and Tsoukalas, Athanasios and Tzes, Anthony},
  journal={arXiv preprint arXiv:2110.08822},
  year={2021}
}
```

#### 致谢
我们的代码基于以下库实现：
* [PyTracking](https://github.com/visionml/pytracking)
* [Stark](https://github.com/researchmm/Stark)
* [Timm](https://github.com/rwightman/pytorch-image-models)

希望这些信息对你有帮助！如果有任何进一步的问题或需要更详细的说明，请随时告知。


# 🧠 Behav-Marker: A Novel Artificial Intelligence based Detection Marker for Prodromal Parkinson's Disease Screening 🔍

This is the code implementation of the method used in our paper 📄 *"Behav-Marker: A Novel Artificial Intelligence based Detection Marker for Prodromal Parkinson's Disease Screening"*

## ⚙️ Requirements
Our models are trained and tested in the environment of Python 3.8, R 4.2.2, PyTorch 1.9.1, CUDA 11.1. 💻 Our interactive software is suitable for win10 and above system versions.

## 📊 Datasets
The third-party datasets used in our study are provided under the Creative Commons Public License 🌐:  
- [Movebank](https://www.movebank.org/)  
- [Microsoft T-Drive Project](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)  
- [Geolife Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)  
- [Taxi GPS Dataset](https://tianchi.aliyun.com/dataset/94216)  
- [Gowalla Dataset](https://snap.stanford.edu/data/loc-gowalla.html).  

### 📂 Data Structure
**Please Note:** 🧪 This study uses the Reworded Open Field 63007 experimental device (RWD Life Science Co., Ltd., Shenzhen, China). If you are using a different device for data collection, please make sure to preprocess your data into the same style as used in this project (Including but not limited to the color, shape of the image). ⚠️ Otherwise, the provided models may not work directly with your data — you may need to fine-tune or retrain the models to ensure compatibility. 🔧📊

We make internal data public 📢. The internal data (trajectory plots, heatmaps and motor behavior characteristics) used in this study and the open field test features (Total Movement Distance, Average Velocity, Line Crossing Frequency, Center Zones Entries, Center Zone Duration, Center Zone Path Length, Peripheral Zones Entries, Peripheral Zone Duration, Peripheral Zones Path Length) are now open to the research community through an open platform🤗 ([PD-Movement-Behavior-Dataset](https://huggingface.co/datasets/WeiWei-XPU/PD-Movement-Behavior-Dataset)) 🤝, which can be used by academic research peers to verify experimental results, promote method replication, and promote collaborative exploration in the field of early screening for Parkinson's disease.

Dataset with the following folder structure 📁:
```
Dataset/
├──Characteristics of open field test/
│   ├──Control.csv
│   ├──Third.csv
│   ├──Sixth.csv
│   ├──Tenth.csv
├── heat map/
│   ├── Control/
│   ├── Third/
│   ├── Sixth/
│   └── Tenth/
└── Trajectories/
    ├── Control/
    ├── Third/
    ├── Sixth/
    └── Tenth/
 ```

## 🛠️ Open-Source Parkinson's Disease Screening Tool (v1.0.0)
In order to automate the screening and analysis of prodromal behaviors of Parkinson's disease 🤖, we have developed an interactive software tool 🖥️ (applicable to Windows 10 and above) that integrates the Parkinson's disease screening model. By simply uploading two types of data 📤, trajectory map and heat map, the risk of Parkinson's disease can be quantitatively assessed. The interactive software is now open to the scientific research community through an open platform 🔓, and can be used by academic and scientific research peers to verify the experimental results.

## 🤖 Models
We also make public the best model parameters optimized on the internal dataset used in this study (.pth format) 🧠, which can be obtained through the open platform:
- [🔗 Multimodal-fusion-ViT-base-model](https://huggingface.co/WeiWei-XPU/Multimodal-fusion-ViT-base)
- [🔗 Foundation-model-heatmap](https://huggingface.co/WeiWei-XPU/foundation-model-heatmap)
- [🔗 Multimodal-fusion-densenet121-model](https://huggingface.co/WeiWei-XPU/Multimodal-fusion-densenet121-model)
- [🔗 Multimodal-fusion-resnet50-model](https://huggingface.co/WeiWei-XPU/Multimodal-fusion-resnet50-model)
- [🔗 Foundation-model-trajectory-plot](https://huggingface.co/WeiWei-XPU/foundation-model-trajectory-plot)
- [🔗 Multimodal-fusion-foundation-model](https://huggingface.co/WeiWei-XPU/Multimodal-fusion-foundation-model)
- [🔗 Multimodal-fusion-ViT-large-model](https://huggingface.co/WeiWei-XPU/Multimodal-fusion-ViT-large-model)
- [🔗 Pretrained-densenet121-heatmap-model](https://huggingface.co/WeiWei-XPU/pretrained-densenet121-heatmap-model)
- [🔗 Pretrained-densenet121-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/pretrained-densenet121-trajectory-plot-model)
- [🔗 Pretrained-resnet50-heatmap-model](https://huggingface.co/WeiWei-XPU/pretrained-resnet50-heatmap-model)
- [🔗 Pretrained-resnet50-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/pretrained-resnet50-trajectory-plot-model)
- [🔗 Pretrained-ViT-base-heatmap-model](https://huggingface.co/WeiWei-XPU/pretrained-ViT-base-heatmap-model)
- [🔗 Pretrained-ViT-base-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/pretrained-ViT-base-trajectory-plot-model)
- [🔗 Pretrained-ViT-large-heatmap-model](https://huggingface.co/WeiWei-XPU/pretrained-ViT-large-heatmap-model)
- [🔗 Pretrained-ViT-large-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/pretrained-ViT-large-trajectory-plot-model)
- [🔗 Random-initial-densenet121-heatmap-model](https://huggingface.co/WeiWei-XPU/random-intial-densenet121-heatmap-model)
- [🔗 Random-initial-densenet121-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/random-intial-densenet121-trajectory-plot-model)
- [🔗 Random-initial-resnet50-heatmap-model](https://huggingface.co/WeiWei-XPU/random-intial-resnet50-heatmap-model)
- [🔗 Random-initial-resnet50-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/random-intial-resnet50-trajectory-plot-model)
- [🔗 Random-initial-ViT-base-heatmap-model](https://huggingface.co/WeiWei-XPU/random-intial-ViT-base-heatmap-model)
- [🔗 Random-initial-ViT-base-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/random-intial-ViT-base-trajectory-plot-model)
- [🔗 Random-initial-ViT-large-heatmap-model](https://huggingface.co/WeiWei-XPU/random-intial-ViT-large-heatmap-model)
- [🔗 Random-initial-ViT-large-trajectory-plot-model](https://huggingface.co/WeiWei-XPU/random-intial-ViT-large-trajectory-plot-model).

## 🛠️ Foundation models fine-tuning
```bash
conda create -n mae_finetune python=3.8
conda activate mae_finetune
```
```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pandas scikit-learn matplotlib opencv-python
```
# 📂Directory structure requirements
```
/data/zhenyuan/dataset
├── train/
│   ├── class0/
│   └── class1/
├── validation/
│   ├── class0/
│   └── class1/
└── test/
    ├── class0/
    └── class1/
```
```bash
# Pretraining weight path
pretrain_path = "/data/zhenyuan/pretrain/save_result/encoder_checkpoint.pth"
```
```bash
# Data path
train_dataset = '/data/zhenyuan/dataset/train'
val_dataset = '/data/zhenyuan/dataset/validation'
test_dataset = '/data/zhenyuan/dataset/test'
```
```bash
# Output path
save_path = "/data/zhenyuan/save_model(outputs)/foundation_model_HM.pth"
```
## 🛠️Fine-tuning execution
```bash
# Basic training (single GPU)
python Heatmap_foundation_model_fine-tuning.py
```
```bash
# Multi-GPU training (2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 Heatmap_foundation_model_fine-tuning.py
```

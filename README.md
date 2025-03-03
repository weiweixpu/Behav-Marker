# Behav-Marker: A Novel Artificial Intelligence based Detection Marker for Prodromal Parkinson's Disease Screening
This is the code implementation of the method used in our paper *"Behav-Marker: A Novel Artificial Intelligence based Detection Marker for Prodromal Parkinson's Disease Screening"*
## Requirements
Our models are trained and tested in the environment of Python 3.8, R 4.2.2, PyTorch 1.9.1, CUDA 11.1. Our interactive software is suitable for win10 and above system versions.
## Datasets
The third-party datasets used in our study are provided under the Creative Commons Public License:  
- [Movebank](https://www.movebank.org/)  
- [Microsoft T-Drive Project](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/)  
- [Geolife Trajectories 1.3](https://www.microsoft.com/en-us/download/details.aspx?id=52367)  
- [Taxi GPS Dataset](https://tianchi.aliyun.com/dataset/94216)  
- [Gowalla Dataset](https://snap.stanford.edu/data/loc-gowalla.html).  
### Data Structure
We make our internal data public. The internal data (trajectory maps and heat maps) used in this study are now open to the research community through an open platform ([PD-Movement-Behavior-Dataset](https://huggingface.co/datasets/WeiWei-XPU/PD-Movement-Behavior-Dataset)), which can be used by academic research peers to verify experimental results, promote method replication, and promote collaborative exploration in the field of early screening for Parkinson's disease.
Datasets with the following folder structure.
```
Dataset/
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
## Open-Source Parkinson's Disease Screening Tool (v1.0.0)
In order to automate the screening and analysis of prodromal behaviors of Parkinson's disease, we have developed an interactive software tool (applicable to Windows 10 and above) that integrates the Parkinson's disease screening model. By simply uploading two types of data, trajectory map and heat map, the risk of Parkinson's disease can be quantitatively assessed. The interactive software is now open to the scientific research community through an open platform (), and can be used by academic and scientific research peers to verify the experimental results.


## Models
We also make public the best model parameters optimized on the internal dataset used in this study (.pth format), which can be obtained through the open platform ().

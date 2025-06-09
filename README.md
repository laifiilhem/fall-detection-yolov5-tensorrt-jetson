# 🛡️ Fall Detection using YOLOv5 and TensorRT on Jetson Nano

Welcome to this real-time **Fall Detection System** powered by **YOLOv5**, trained on a custom dataset and deployed on **Jetson Nano** using **TensorRT** for optimized inference. The goal is to detect whether a person is **standing**, **sitting**, or has **fallen**, in real time, using only a camera.

---

## 📁 Project Structure

fall-detection-yolov5-tensorrt-jetson/
├── models/ # Trained weights (.pt, .wts, .engine)
├── jetson_optimisation/ # TensorRT conversion files
├── notebooks/ # Training logs, evaluation, confusion matrix
├── inference/ # Real-time inference scripts
├── data/ # Dataset (images + labels)
├── README.md # Project description (this file)
└── LICENSE # License (MIT)


---

## 🚀 Step 1: Training on PC

###  Environment Setup (PC)

**(Optional but recommended)** Create a dedicated conda environment:

```bash
conda create -n yoloenv38 python=3.8 -y
conda activate yoloenv38
```

Install PyTorch & torchvision (CUDA 11.8):

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```



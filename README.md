#  Fall Detection using YOLOv5 and TensorRT on Jetson Nano

Welcome to this real-time **Fall Detection System** powered by **YOLOv5**, trained on a custom dataset and deployed on **Jetson Nano** using **TensorRT** for optimized inference. The goal is to detect whether a person is **standing**, **sitting**, or has **fallen**, in real time, using only a camera.

---

## 📁 Project Structure

fall-detection-yolov5-tensorrt-jetson/

├── models/ # Trained weights (.pt, .wts, .engine)

├── jetson_optimisation/ # TensorRT conversion files

├── notebooks/ # Training logs, evaluation, confusion matrix

├── inference/ # Real-time inference scripts

├── datasets/ # Dataset (images + labels)

├── README.md # Project description (this file)

└── LICENSE # License (MIT)


---

##  Step 1: Training on PC

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

Clone YOLOv5 v6.0

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v6.0
pip install -r requirements.txt

```
📁 Prepare Dataset
Configure your data.yaml file like this:

    train: ../data/images/train
    val: ../data/images/val
    test: ../data/images/test

    nc: 3
    names: ['fall-action', 'sit', 'stand']

Train the Model

```bash
python train.py --img 640 --batch 16 --epochs 500 --data data.yaml \
  --weights yolov5s.pt --name fall_yolo
```
 Evaluation Notebooks
 
 notebooks/01_training_metrics.ipynb → View results.csv

notebooks/02_test_inference.ipynb → Evaluate predictions on test set

notebooks/03_confusion_matrix.ipynb → Display and save confusion matrix

##  Step 2: Deployment on Jetson Nano
###  Environment Setup (Jetson)
JetPack 4.6 (includes CUDA 10.2, cuDNN, TensorRT)

Python 3.6 is used by default on Jetson Nano

 Install PyTorch & torchvision on Jetson Nano
 Use the official NVIDIA wheel for Jetson Nano:

 ```bash
# PyTorch
wget https://nvidia.box.com/shared/static/p57jwntlpb2qk1n52f7v5c6hrp9ddp4f.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# torchvision
sudo apt-get install libjpeg-dev zlib1g-dev
git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
 ```
### Install Other Dependencies
```bash
pip3 install numpy==1.19.0 pandas Pillow tqdm imutils pycuda seaborn

```

## Step 3: Convert Model to TensorRT Engine
### Generate .wts and .engine Files
```bash
cd jetson_optimisation/tensorrtx/yolov5
python3 gen_wts.py -w ../../../models/best.pt -o best.wts

mkdir build && cd build
cmake ..
make
./yolov5 -s best.wts best.engine s
```
Output: best.engine file (TensorRT serialized engine)



## Step 4: Real-Time Inference
Use webcam, RTSP stream, or video file:

bash
Copier le code

```bash
cd inference
python3 video_inference.py

```
Detections are displayed with bounding boxes and labels: fall-action, sit, stand.

## ✅ Results & Performance

mAP, precision, recall, and confusion matrix generated during training

Inference speed on Jetson Nano with TensorRT is significantly faster


## License
This project is licensed under the MIT License.

## Author
Ilhem Laifi
Electrical Engineering Student – ENIT, Tunisia

##  Acknowledgments
Ultralytics YOLOv5

TensorRTx by wang-xinyu

NVIDIA Jetson Forum – PyTorch

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Thank you for visiting this repository! ⭐





















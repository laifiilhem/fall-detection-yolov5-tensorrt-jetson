#  Trained Models – Fall Detection 

This folder contains the exported and optimized models used in the Fall Detection Final Year Project (PFA) developed by **Ilhem** at ENIT.

---

## 📁 Contents

- `best.pt`  
  - Format: PyTorch (`.pt`)  
  - Description: YOLOv5 model trained on the Fall Action Dataset using YOLOv5 v6.0 (Ultralytics).
  - Usage: Can be loaded using the YOLOv5 PyTorch repo for inference or further training.

- `best.wts`  
  - Format: Weights (`.wts`)  
  - Description: Converted model weights for TensorRT conversion using `yolov5/build_engine.py`.
  - Usage: Intermediate file used to generate `.engine` optimized model for Jetson Nano.

- `best.engine`  
  - Format: TensorRT Engine (`.engine`)  
  - Description: Final optimized model deployed on Jetson Nano for real-time fall detection.
  - Usage: Inference with TensorRT Python/C++ bindings on Jetson devices.

---

## ⚠️ Notes

- The `.pt` model was trained on a custom dataset composed of fall, sit, and stand images using YOLOv5 v6.0.
- The `.wts` and `.engine` files were generated to accelerate inference on NVIDIA Jetson Nano using TensorRT.
- The model predicts 3 classes:
  - `0`: Fall
  - `1`: Sit
  - `2`: Stand

---

## 🧠 YOLOv5 Training Info

- YOLO version: `v6.0`  
- Input size: `640x640`  
- Batch size: `16`  
- Epochs: `100`

---

## 👤 Author

**Ilhem**  
Electrical Engineering Student, ENIT  
🔗 [LinkedIn](https://www.linkedin.com/in/ilhem-laifi-9682702b2/)  
💻 [GitHub](https://github.com/laifiilhem)

---


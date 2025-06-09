# Fall Action Dataset – Human Fall Detection

## 🎯 Objective

This dataset is designed for developing and evaluating human fall detection systems using computer vision. It contains a wide variety of human positions captured under different angles and environmental conditions, making it suitable for training robust deep learning models.

>  This dataset was merged, cleaned, and re-annotated by **Ilhem Laifi** during her final year project (PFA) at ENIT, focused on **human fall detection using YOLOv5 and deployment on Jetson Nano**.

---

## 🗂️ Dataset Structure

- **Total images:** 4116  
  - `train/`: 3568 images  
  - `valid/`: 396 images  
  - `test/`: 152 images  
- **Annotation format:** YOLO TXT files  
- **Classes:**
  - `0`: Fall-Action
  - `1`: Sit
  - `2`: Stand

---

##  Dataset Composition

This dataset is a cleaned and unified combination of two publicly available datasets:

1. [Fall Detect2 (Roboflow)](https://universe.roboflow.com/goldfish/fall-detect2/dataset/11)  
   - License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
   - Source: Roboflow Universe  

2. [Fall Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset)  
   - License: [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)  
   - Source: Kaggle  

**Cleaning & Annotation Notes:**
- Removed visually ambiguous or irrelevant frames (e.g., non-fall lying positions).
- Verified and normalized all bounding boxes.
- Unified class naming and numbering.

---

##  Use Cases

- Fall detection systems for elderly care  
- Smart surveillance in homes and hospitals  
- Human activity recognition models  
- Training AI pose classifiers

---

##  Format Details

- **Image types:** `.jpg`, `.png`  
- **Annotation format:** YOLOv5 compatible `.txt` files (normalized coordinates)  
- **Image sizes:** Mixed resolutions

---

##  Licensing & Credits

This dataset is built upon open datasets licensed under:

- [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/)

If you use this dataset in academic work or public projects, please cite the original datasets and credit the re-annotation effort.

---

##  Author

**Ilhem**  
Electrical Engineering Student, ENIT  
🔗 [LinkedIn](https://www.linkedin.com/in/ilhem-laifi-9682702b2/)  
💻 [GitHub](https://github.com/laifiilhem)

---

## 💬 Contact

Feel free to open an issue or contact me for collaboration or questions related to this dataset.

---

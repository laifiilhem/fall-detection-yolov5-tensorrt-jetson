# Notebooks for Fall Detection PFA

This folder contains Jupyter notebooks and related data to analyze and evaluate the YOLOv5 fall detection model (`best.pt`).

## Contents

- `training_metrics.ipynb`  
  Analyze training and validation metrics (precision, recall, mAP, losses) from `rsults.csv`.

- `test_inference.ipynb`  
  Run inference on a set of unseen test images using the trained model `best.pt`. Outputs saved in `outputs/`.

- `confusion_matrix_analysis.ipynb`  
  Generate and visualize the confusion matrix from test predictions to evaluate classification accuracy across three classes:
  - Fall-Action (0)  
  - Sit (1)  
  - Stand (2)  

- `results.csv`  
  Training and validation metrics exported during model training.

- `outputs/`  
  Contains:
  - Confusion matrix plots  
  - Test images with predicted bounding boxes  

## Usage

1. Start with `training_metrics.ipynb` to check the model's training progress.  
2. Run `test_inference.ipynb` to test the model on new images.  
3. Use `confusion_matrix_analysis.ipynb` to assess model performance via the confusion matrix.

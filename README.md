## 🩺 Chest X-Ray Disease Classification  

This repository contains a Convolutional Neural Network (CNN) baseline model for classifying chest X-ray images into three classes:
- **Normal**
- **Pneumonia**
- **Tuberculosis**

Dataset: https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset

The project includes:

- A clean training notebook

- Modular Python scripts (src/)

- Grad-CAM visualization

### 📂 Repository Structure

                                chest-xray-classifier/
                                │
                                ├── README.md
                                ├── requirements.txt
                                │
                                ├── src/
                                │   ├── train.py
                                │   ├── model.py
                                │   ├── utils.py
                                │   └── chest_xray_notebook.ipynb   ← FULL TRAINING NOTEBOOK
                                │
                                └── data/   ← (user must add dataset here)


### 📥 Dataset Setup
You must download the Chest X-Ray Pneumonia dataset (Kaggle) and place it like this:

                                data/
                                │
                                ├── train/
                                │   ├── normal/
                                │   └── pneumonia/
                                │   └── tuberculosis/
                                │
                                ├── val/
                                │   ├── normal/
                                │   └── pneumonia/
                                │   └── tuberculosis/
                                │
                                └── test/
                                            ├── normal/
                                            └── pneumonia/
                                            └── tuberculosis/

## 🚀 How to Use This Project
Use the Jupyter Notebook located in: src/chest_xray_notebook.ipynb

🎓 The Notebook Includes:
  ✔ Data loading
  ✔ Exploratory plots
  ✔ Sample images from each class
  ✔ Model creation
  ✔ Model training
  ✔ Evaluation
  ✔ Grad-CAM heatmaps

▶ How to run

  pip install -r requirements.txt

  jupyter notebook src/chest_xray_notebook.ipynb


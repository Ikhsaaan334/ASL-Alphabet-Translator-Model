# 🤟 ASL Alphabet Translator Model

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?logo=tensorflow&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.2-76B900?logo=nvidia&logoColor=white)
![cuDNN](https://img.shields.io/badge/cuDNN-8.1-76B900?logo=nvidia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview
**ASL Alphabet Translator Model** is a Deep Learning project designed to recognize and translate American Sign Language (ASL) alphabets from images or real-time video streams. The model utilizes Convolutional Neural Networks (CNN) to classify hand gestures into their corresponding letters (A-Z) along with generic gestures like 'space' or 'del'.

This project aims to bridge the communication gap by providing a digital tool to interpret sign language efficiently using GPU-accelerated computing.

## ✨ Features
* **High Accuracy:** Trained on a comprehensive dataset of ASL hand gestures.
* **Real-time Prediction:** Capable of predicting gestures from a webcam feed.
* **GPU Acceleration:** Optimized for NVIDIA GPUs using CUDA 11.2 and cuDNN 8.1.
* **Visualizations:** Includes training metrics (Accuracy & Loss graphs).

## 📂 Dataset
The model was trained using the **ASL Alphabet Dataset**.
* **Training Images:** 87,000 images (29 classes)
* **Image Size:** 200x200 pixels
* **Classes:** A-Z, space, del, nothing

## ⚙️ Environment Setup

To run this project, please ensure your environment matches the specific versions below to avoid compatibility issues with TensorFlow and GPU drivers:

* **Python:** 3.10
* **TensorFlow:** 2.10.0
* **CUDA Toolkit:** 11.2
* **cuDNN:** 8.1

### Installation Steps

**Clone the repository**
```
git clone [https://github.com/Ikhsaaan334/ASL-Alphabet-Translator-Model.git](https://github.com/Ikhsaaan334/ASL-Alphabet-Translator-Model.git)
cd ASL-Alphabet-Translator-Model
```

***Create a Virtual Environment (Optional but recommended)***
```
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

***Training the Model***
To train the model from scratch, run the training script
```
python train_model.py
```

***Testing / Prediction***
```
python predict.py
```

***Model Performance***
Current model metrics:
<img width="1200" height="500" alt="training_history" src="https://github.com/user-attachments/assets/8b2e4f05-d6eb-451f-b959-163e9a8bf7d1" />



# 🧠 DA6401 Assignment: Image Classification & Object Detection
This project is divided into three parts:
- Part A: Build a Convolutional Neural Network (CNN) from scratch.
- Part B: Fine-tune Pretrained Models (e.g., Inception, ResNet).
- Part C: Use a Pretrained YOLOv5 model for object detection.

## 🔧 1️⃣ Setup Instructions
- Ensure you have Conda installed.
- Create an environment and install dependencies.
```bash
conda env create -n CNN python=3.10
conda activate CNN 
pip install -r requirements.txt
```


## 2️⃣ Part A : Implementing CNN from Scratch
Train and tune a CNN model on the iNaturalist dataset.
### 🔁 Run a W&B Sweep
```bash
wandb sweep sweep.yaml
wandb agent <your-entity>/<project-name>/<sweep-id>
```
### 📊 Visualize the results
```bash
python visualise.py
```

## 🧠 3️⃣ Part B: Pretrained Models (Transfer Learning)

```bash
python main.py --strategy=last_k --k=3 
```
Available Strategies:
- last_k : FInetune last k layers.
- last : Finetune just the lat layer.
- full : Train full model

## 4️⃣ Part C : Object Detection using Pretrained YOLOV
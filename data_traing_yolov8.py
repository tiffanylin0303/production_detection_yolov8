from ultralytics import YOLO
import os
import torch

os.environ['CUDA_VISIBLE_DEVICE']='1'
print(torch.cuda.device_count())


# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
print(model.info())
def train_model():

    # Train the model 
    results = model.train(
        data = "dataset/data.yaml", 
        epochs = 100, 
        imgsz = 640,
        batch = 16,
        lr0 = 0.01,
        lrf = 0.01, 
        box = 8,
        dfl = 2
        )


if __name__ == '__main__':
    train_model()
  
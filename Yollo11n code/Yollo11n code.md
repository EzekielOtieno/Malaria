```python
mport os
from ultralytics import YOLO

# Step 1: Set paths to your local files
data_yaml_path = r"C:\Users\Big Data Team\Downloads\complete_data\complete_data\data.yaml"
model_path = r"C:\Users\Big Data Team\Downloads\yolo11n-seg.pt"  # Path to the YOLO model

# Step 2: Load the YOLO model
model = YOLO(model_path)

# Step 3: Train the model
epochs = 100  # Number of epochs

results = model.train(
    data=data_yaml_path,
    epochs=epochs,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size
    device='cuda',  # Use 'cuda' for GPU, or 'cpu' for CPU
    project='E:/model',  # Directory to save the results
    name='trained_model'  # Name of the training session
)

print("Training completed successfully.")

```

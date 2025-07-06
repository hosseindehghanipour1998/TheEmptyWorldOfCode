import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image
import warnings

# Suppress a common PIL warning when loading images
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Ensure you have the yolov5 package installed: pip install yolov5
# Or if you prefer using a direct download (less recommended for dependency management):
# git clone https://github.com/ultralytics/yolov5
# pip install -r yolov5/requirements.txt


# --- 1. Model Definition ---
class AttentionModule(nn.Module):
    """
    A simple example of a Channel Attention Module (like a simplified SE-Net).
    You can replace this with more complex attention mechanisms (e.g., CBAM, Self-Attention).
    """
    def __init__(self, channels, reduction_ratio=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        channel_attention = self.fc2(self.relu(self.fc1(avg_pool)))
        return x * self.sigmoid(channel_attention)


class YOLOClassificationModel(nn.Module):
    def __init__(self, num_classes, pretrain_path='yolov5s.pt', freeze_backbone=True):
        super(YOLOClassificationModel, self).__init__()
        print(f"Loading YOLOv5s model from {pretrain_path}...")
        try:
            # Load pre-trained YOLOv5 model
            # This loads the entire model, we'll extract the backbone
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            print("YOLOv5s model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv5s model: {e}")
            print("Please ensure you have an internet connection or the yolov5s.pt file locally.")
            print("You might need to install `yolov5` package: `pip install yolov5`")
            raise

        # Extract the backbone (features part of YOLOv5)
        # YOLOv5's model structure usually has 'model.0' as the backbone
        # We need to find the correct module that represents the feature extractor.
        # For yolov5s, 'model.model' usually contains the full sequential model,
        # and we want to remove the detection head.
        # A common approach is to take up to the last pooling/detection layer.
        # Let's inspect the model structure (uncomment if you want to see it):
        # print(self.yolo_model)
        
        # A simplified way to get the backbone for classification is to take
        # the feature extractor part up to the last convolutional layer before the detection head.
        # The YOLOv5 'model' attribute holds the nn.Module.
        # We'll re-purpose its 'model' part as our backbone.
        # The detection head typically comes after the 'Detect' layer.
        # We will iterate through modules and take up to the 'Detect' layer or a common feature map output.
        
        # In YOLOv5, the last feature map before the detection head often comes from layer 23 or 24 (depending on architecture).
        # We'll assume the backbone ends before the detection head which is usually 'model.24' for yolov5s.
        # A more robust way might involve tracing or knowing the exact architecture.
        # For yolov5s, the model is structured into a C3 module followed by a detection head.
        # We can directly use the `model.model` and slice it or just pick the feature extractor part.
        
        # Let's use the `model` attribute directly which is a nn.Sequential module
        # and remove its last layer (the detection head).
        # Note: This is an empirical approach. For more complex modifications,
        # you might need to dive into YOLOv5's `models/yolo.py` for exact layer indices.
        
        # A more robust way: use model.model and identify where the backbone ends.
        # For YOLOv5s, a common output for features used for classification would be from the penultimate layer
        # before the final detection head branches.
        
        # Let's try to get the 'model.model' part which is the nn.Sequential backbone
        self.backbone = self.yolo_model.model[:-1] # Remove the last layer which is the Detect head

        if freeze_backbone:
            print("Freezing YOLO backbone layers...")
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine the number of output channels from the YOLO backbone.
        # This depends on the last layer of the backbone.
        # For yolov5s, the output of the last backbone layer is often 1024 channels (C3 module output).
        # We'll perform a dummy forward pass to get the shape if uncertain.
        # If running on CPU, dummy pass might take time or fail if the model expects GPU.
        
        # Let's assume the last layer of the backbone outputs 1024 channels for yolov5s.
        # This is based on typical YOLOv5 architecture for 's' model.
        # If your model variant is different, you might need to adjust `backbone_out_channels`.
        # You can inspect `self.yolo_model.model[-2]` and its `c2` attribute for exact output channels.
        backbone_out_channels = 1024 
        
        # Add attention layers
        self.attention = AttentionModule(channels=backbone_out_channels) # Apply attention after backbone

        # Global average pooling to flatten features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels, num_classes) # Input to linear layer should match attention output
        )

    def forward(self, x):
        # Pass through the YOLO backbone
        features = self.backbone(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling
        pooled_features = self.avgpool(attended_features)
        
        # Classify
        output = self.classifier(pooled_features)
        return output

# --- 2. Data Module ---
class CustomDatasetModule:
    def __init__(self, data_dir, image_size=(640, 640), batch_size=32, val_split=0.2):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.class_names = []
        self.num_classes = 0
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None # You might want a separate test set

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._prepare_datasets()

    def _prepare_datasets(self):
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
        self.class_names = full_dataset.classes
        self.num_classes = len(self.class_names)
        print(f"Detected classes: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")

        # Split into training and validation sets
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1)

# --- 3. Training Function ---
def train_model(model, data_module, num_epochs=10, learning_rate=0.001, device='cpu', checkpoint_path='best_model.pth'):
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Learning rate scheduler

    model.to(device)
    best_val_accuracy = 0.0

    print(f"\nStarting training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = val_correct_predictions / val_total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with validation accuracy: {best_val_accuracy:.4f}")
            
        scheduler.step() # Update learning rate

    print("Training finished!")

# --- 4. Prediction Function (for inference after training) ---
def predict(model, image_path, class_names, device='cpu', image_size=(640, 640)):
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted_class_idx = torch.max(probabilities, 1)
            
            predicted_label = class_names[predicted_class_idx.item()]
            confidence = probabilities[0, predicted_class_idx.item()].item()
            
            return predicted_label, confidence
    except Exception as e:
        print(f"Error during prediction for {image_path}: {e}")
        return None, None

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_DIR = "path/to/your/image_dataset" # <--- IMPORTANT: Change this to your dataset path
    # Your dataset structure should be:
    # image_dataset/
    # ├── plastic/
    # │   ├── img1.jpg
    # │   └── img2.png
    # ├── metal/
    # │   ├── imgA.jpeg
    # │   └── imgB.jpg
    # └──

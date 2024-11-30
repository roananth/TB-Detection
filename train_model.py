import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
from torchvision import datasets, models, transforms


# Set device, use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations for both training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Define the path to the dataset directories
train_dir = './train'
val_dir = './val'

# Create datasets for training and validation
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_data = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

# Set batch size and create data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the model (using ResNet18 as an example)
class ClassifyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassifyModel, self).__init__()
        self.model = models.resnet152(pretrained=True)
        # Replace the final layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Instantiate the model and move it to the appropriate device
model = ClassifyModel(num_classes=2).to(device)
model.train()  # Set the model to training mode

# Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Defining Scheduler    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Ensure the model is in training mode
    running_loss = 0.0

    # Training phase
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate the average loss for this epoch
    train_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    # Validation phase (optional but recommended)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy * 100:.2f}%")

# Save the trained model, ensuring the directory exists
os.makedirs('./model', exist_ok=True)
torch.save(model.state_dict(), './model/Resnet-ClassifyModel.pth')
print("Model training completed and saved successfully.")
import torch
from torch.utils.data import DataLoader
from train_model import ClassifyModel  
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transformations
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Set the path to the validation directory
val_dir = './val'

# Create validation dataset
val_data = datasets.ImageFolder(val_dir, transform=data_transforms['val'])

# Set the batch size and create a data loader
batch_size = 16
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Load the pre-trained ResNet model
model = ClassifyModel(num_classes=2)
model = model.to(device)

# Load the trained model
model.load_state_dict(torch.load('./model/Resnet-ClassifyModel.pth'))
model.eval()  # Set the model to evaluation mode

# Define class names
class_names = val_data.classes
print(f"Class names: {class_names}")

# # Validation loop
# correct_predictions = 0
# total_images = 0
# true_positives = 0
# false_negatives = 0

# with torch.no_grad():
#     # Iterate through the validation images
#     for images, labels in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)

#         # Process each image in the batch
#         for i in range(images.size(0)):
#             image_path = val_data.imgs[total_images + i][0]
#             predicted_label = predicted[i].item()
#             true_label = labels[i].item()

#             # Calculate accuracy and recall
#             if predicted_label == true_label:
#                 correct_predictions += 1
#                 if true_label == class_names.index('Y'):
#                     true_positives += 1
#             elif true_label == class_names.index('Y'):
#                 false_negatives += 1

#             # Print the predicted label with the complete image path
#             print(f"Image: {image_path}, Predicted Label: {class_names[predicted_label]}")

#         total_images += images.size(0)

# # Calculate accuracy and recall
# accuracy = correct_predictions / total_images
# recall = true_positives / (true_positives + false_negatives)

# # Print accuracy and recall
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Recall: {recall:.4f}")

# Load the pre-trained model and move it to the device
model = ClassifyModel(num_classes=2)
model.load_state_dict(torch.load('./model/Resnet-ClassifyModel.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

def calculate_metrics(predictions, true_labels):
    """Calculate accuracy, precision, recall, and F1-score."""
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    precision = precision_score(true_labels, predictions, pos_label=class_names.index('Y'))
    recall = recall_score(true_labels, predictions, pos_label=class_names.index('Y'))
    f1 = f1_score(true_labels, predictions, pos_label=class_names.index('Y'))
    return accuracy, precision, recall, f1

# Validation loop
all_predictions = []
all_true_labels = []
total_images = len(val_data)

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Store predictions and true labels for metric calculation
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

        # Print each image's prediction
        for i in range(images.size(0)):
            image_path = val_data.imgs[total_images - len(all_predictions) + i][0]
            print(f"Image: {image_path}, Predicted Label: {class_names[predicted[i].item()]}")

# Calculate and print final metrics
accuracy, precision, recall, f1 = calculate_metrics(all_predictions, all_true_labels)
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1-Score: {f1:.4f}")

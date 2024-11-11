# Set device
device = torch.device('cuda')

# Define data transformations
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Set the path to the validation directory
val_dir = './data/val'

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

# Validation loop
correct_predictions = 0
total_images = 0
true_positives = 0
false_negatives = 0

with torch.no_grad():
    # Iterate through the validation images
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Process each image in the batch
        for i in range(images.size(0)):
            image_path = val_data.imgs[total_images + i][0]
            predicted_label = predicted[i].item()
            true_label = labels[i].item()

            # Calculate accuracy and recall
            if predicted_label == true_label:
                correct_predictions += 1
                if true_label == class_names.index('Y'):
                    true_positives += 1
            elif true_label == class_names.index('Y'):
                false_negatives += 1

            # Print the predicted label with the complete image path
            print(f"Image: {image_path}, Predicted Label: {class_names[predicted_label]}")

        total_images += images.size(0)

# Calculate accuracy and recall
accuracy = correct_predictions / total_images
recall = true_positives / (true_positives + false_negatives)

# Print accuracy and recall
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")

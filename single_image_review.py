import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
model = torch.load("our_model.pth")  # Adjust path to our saved model
model.eval()  # Set the model to evaluation mode

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),         # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats or your training dataset stats
                         std=[0.229, 0.224, 0.225])
])

# Load the image
image_path = "path_to_image.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB format
input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    tb_presence_score = probabilities[1].item()  # Assuming class 1 is "TB Present"

# Use a threshold (e.g., 0.5) to classify
if tb_presence_score > 0.5:
    print(f"TB is likely present with a confidence score of {tb_presence_score:.2f}.")
else:
    print(f"TB is unlikely with a confidence score of {tb_presence_score:.2f}.")
    

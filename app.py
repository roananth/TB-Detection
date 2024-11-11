from flask import Flask, render_template, request, jsonify
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from classify_model import ClassifyModel
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClassifyModel(num_classes=2).to(device)
model.load_state_dict(torch.load('./model/Resnet-ClassifyModel.pth'))
model.eval()  # Set model to evaluation mode

# Define data transformation for uploaded images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict function to handle the image input and return result
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    # Return the predicted class (0 or 1), where 0 = "No tuberculosis", 1 = "Tuberculosis"
    return predicted.item()

# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Make prediction
    prediction = predict_image(file_path)
    
    # Map the prediction to the actual class name
    result = 'Tuberculosis' if prediction == 1 else 'No Tuberculosis'

    # Return the result to the user
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)

import os
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Define allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(224),  # Crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    
    image = Image.open(image_path)
    
    # Convert grayscale images to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Function to handle image prediction
def predict_image(image_path, model):
    image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

class ClassifyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ClassifyModel, self).__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load your model
model_path = 'model/Resnet-ClassifyModel.pth'
model = ClassifyModel(num_classes=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading a chest scan
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            
            # Predict the image class
            predicted_class = predict_image(file_path, model)
            prediction_text = (
                'No signs of tuberculosis detected'
                if predicted_class == 0
                else 'Signs of tuberculosis detected'
            )
            
            # Store prediction in session
            session['result'] = prediction_text
            session['image_url'] = file_path
            
            return redirect(url_for('result'))
    return render_template('upload.html')

# Route for showing prediction results
@app.route('/result')
def result():
    result_data = {
        'prediction': session.get('result', 'No prediction available'),
        'image_url': session.get('image_url')
    }
    return render_template('result.html', result=result_data)

if __name__ == '__main__':
    app.run(debug=True)

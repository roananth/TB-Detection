{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2f8eb2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torchvision import transforms\n",
    "from PIL import Image\n",
    "\n",
    "# Load trained model\n",
    "model = torch.load(\"our model path.pth\")  # Adjust path to our saved model\n",
    "model.eval()  # Set the model to evaluation mode\n",
    "\n",
    "# Define preprocessing pipeline\n",
    "preprocess = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),  # Resize to 224x224\n",
    "    transforms.ToTensor(),         # Convert image to tensor\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats or your training dataset stats\n",
    "                         std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "# Load the image\n",
    "image_path = \"path_to_image.jpg\"  # Replace with the path to your image\n",
    "image = Image.open(image_path).convert(\"RGB\")  # Ensure it's in RGB format\n",
    "input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension\n",
    "\n",
    "\n",
    "# Perform inference\n",
    "with torch.no_grad():\n",
    "    output = model(input_tensor)\n",
    "    probabilities = torch.nn.functional.softmax(output[0], dim=0)\n",
    "    tb_presence_score = probabilities[1].item()  # Assuming class 1 is \"TB Present\"\n",
    "    \n",
    "\n",
    "# Use a threshold (e.g., 0.5) to classify\n",
    "if tb_presence_score > 0.5:\n",
    "    print(f\"TB is likely present with a confidence score of {tb_presence_score:.2f}.\")\n",
    "else:\n",
    "    print(f\"TB is unlikely with a confidence score of {tb_presence_score:.2f}.\")\n",
    "    \n",
    "# Would still need to do some error handling within this code. \n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

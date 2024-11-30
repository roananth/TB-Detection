import os
import random
import shutil
import argparse
import pandas as pd


# Load the CSV
csv_file = "shenzhen_metadata.csv"  # Update this to your actual CSV file path
df = pd.read_csv(csv_file)

# Ensure folder paths are correct
base_image_dir = 'images'  # Update this path if needed
os.makedirs(os.path.join(base_image_dir, 'Y'), exist_ok=True)
os.makedirs(os.path.join(base_image_dir, 'N'), exist_ok=True)

# Iterate through each row in the CSV
for index, row in df.iterrows():
    image_name = row['study_id']  # Use the full filename from the CSV
    finding = row['findings']
    
    # Construct the source path
    src = os.path.join(base_image_dir, image_name)

    # Construct the destination folder based on the findings
    if finding == 'Y':
        dst = os.path.join(base_image_dir, 'Y', image_name)
    else:
        dst = os.path.join(base_image_dir, 'N', image_name)

    # Debugging: Print the source path to verify
    print(f"Checking for image: {src}")

    # Move the file to the respective folder if it exists
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {image_name} to {dst}")
    else:
        print(f"Image {image_name} not found.")

print("Image classification completed.")

def split_dataset(dataset_dir, val_ratio, csv_file):
    # Read the CSV to ensure correct file handling
    df = pd.read_csv(csv_file)

    # Create train/val directories
    for category in ['Y', 'N']:
        os.makedirs(f'train/{category}', exist_ok=True)
        os.makedirs(f'val/{category}', exist_ok=True)

        category_path = os.path.join(dataset_dir, category)
        images = os.listdir(category_path)

        # Shuffle the images to ensure random splitting
        random.shuffle(images)

        # Split the dataset based on the validation ratio
        split_index = int(len(images) * (1 - val_ratio))
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Move files to their respective train/val folders
        for image in train_images:
            shutil.move(os.path.join(category_path, image), f'train/{category}/{image}')

        for image in val_images:
            shutil.move(os.path.join(category_path, image), f'val/{category}/{image}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train/val')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation data')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('dataset', type=str, help='Path to the dataset folder')

    args = parser.parse_args()

    split_dataset(args.dataset, args.val_ratio, args.csv_file)

    
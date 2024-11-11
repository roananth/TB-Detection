import os
import random
import shutil
import argparse

# Create command line arguments
parser = argparse.ArgumentParser(description='Move some samples from train to val')
parser.add_argument('--name', required=True, help='Dataset name')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Dataset percentage to move to val')
args = parser.parse_args()

# Dataset path
dataset_name = args.name
dataset_path = f'/datasets/{dataset_name}'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')

# Create val folder
os.makedirs(val_path, exist_ok=True)
os.makedirs(os.path.join(val_path, 'Y'), exist_ok=True)
os.makedirs(os.path.join(val_path, 'N'), exist_ok=True)

# Calculate validation set size
val_ratio = args.val_ratio  # Validation set ratio
val_size_Y = int(len(os.listdir(os.path.join(train_path, 'Y'))) * val_ratio)
val_size_N = int(len(os.listdir(os.path.join(train_path, 'N'))) * val_ratio)

# Randomly select validation samples
random.seed(42)
val_samples_Y = random.sample(os.listdir(os.path.join(train_path, 'Y')), val_size_Y)
val_samples_N = random.sample(os.listdir(os.path.join(train_path, 'N')), val_size_N)

# Move validation samples to val folder
for sample in val_samples_Y:
    src_path = os.path.join(train_path, 'Y', sample)
    dst_path = os.path.join(val_path, 'Y', sample)
    shutil.move(src_path, dst_path)

for sample in val_samples_N:
    src_path = os.path.join(train_path, 'N', sample)
    dst_path = os.path.join(val_path, 'N', sample)
    shutil.move(src_path, dst_path)

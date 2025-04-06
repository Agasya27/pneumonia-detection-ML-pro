import os
import shutil
from pathlib import Path

def setup_directories():
    """Create necessary directories for the dataset."""
    directories = [
        'data/train/NORMAL',
        'data/train/PNEUMONIA',
        'data/val/NORMAL',
        'data/val/PNEUMONIA',
        'data/test/NORMAL',
        'data/test/PNEUMONIA'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def main():
    print("Setting up directory structure...")
    setup_directories()
    
    print("\nIMPORTANT: Please follow these steps EXACTLY to prepare the dataset:")
    print("\n1. Download the dataset from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Extract the downloaded zip file - you should see a folder named 'chest_xray'")
    print("3. Inside the 'chest_xray' folder, you should find:")
    print("   - train/")
    print("   - test/")
    print("   - val/")
    print("\n4. Copy the contents as follows:")
    print("   FROM chest_xray/train/NORMAL/* TO data/train/NORMAL/")
    print("   FROM chest_xray/train/PNEUMONIA/* TO data/train/PNEUMONIA/")
    print("   FROM chest_xray/test/NORMAL/* TO data/test/NORMAL/")
    print("   FROM chest_xray/test/PNEUMONIA/* TO data/test/PNEUMONIA/")
    print("   FROM chest_xray/val/NORMAL/* TO data/val/NORMAL/")
    print("   FROM chest_xray/val/PNEUMONIA/* TO data/val/PNEUMONIA/")
    print("\nNOTE: Make sure to copy the actual image files, not just the folders!")
    print("\nAfter copying, each directory should contain .jpeg files.")
    print("For example, data/train/NORMAL/ should contain files like:")
    print("   - IM-0115-0001.jpeg")
    print("   - IM-0117-0001.jpeg")
    print("   etc.")

if __name__ == "__main__":
    main() 
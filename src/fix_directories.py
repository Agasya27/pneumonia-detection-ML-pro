import os
import shutil

def fix_nested_directories(base_path):
    # List of dataset splits
    splits = ['train', 'val', 'test']
    # List of classes
    classes = ['NORMAL', 'PNEUMONIA']
    
    for split in splits:
        split_path = os.path.join(base_path, 'data', split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist")
            continue
            
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
                
            nested_path = os.path.join(class_path, class_name)
            if os.path.exists(nested_path):
                print(f"Found nested directory in {class_path}")
                # Move all files from nested directory to parent directory
                for filename in os.listdir(nested_path):
                    src = os.path.join(nested_path, filename)
                    dst = os.path.join(class_path, filename)
                    print(f"Moving {src} to {dst}")
                    shutil.move(src, dst)
                # Remove the empty nested directory
                os.rmdir(nested_path)
                print(f"Removed empty nested directory: {nested_path}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    fix_nested_directories(current_dir)
    print("Directory structure fixed successfully!") 
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import layers, models, optimizers

# Uncomment the path based on the dataset you are working with.
data_path = 'Species'
# data_path = 'Varieties'
# data_path = 'Impurties'

image_size = (224, 224)  # Resize images to this size for ResNet50
batch_size = 32

root_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))  # Get the absolute path of the project root folder
data_folder = os.path.join(root_folder, 'Data', data_path)  # Folder containing the dataset
# Get the class names based on the folder names
class_names = sorted(os.listdir(data_folder))

# Load image file paths and corresponding labels
image_paths = []
labels = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(data_folder, class_name)
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        image_paths.append(image_path)
        labels.append(class_index)

# Split data into training (80%) and testing (20%) sets
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2,
                                                                      random_state=42)

# Step 6: K-Fold Cross-Validation
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(k_fold.split(train_paths, train_labels)):
    print(f"Training fold {fold + 1}...")

    # Generate training and validation data for this fold
    train_paths_fold = np.array(train_paths)[train_idx]
    train_labels_fold = np.array(train_labels)[train_idx]
    val_paths_fold = np.array(train_paths)[val_idx]
    val_labels_fold = np.array(train_labels)[val_idx]

    # Load and preprocess the training and validation images
    train_data_gen = ImageDataGenerator()
    train_data = train_data_gen.flow_from_directory(data_folder, target_size=image_size, batch_size=batch_size,
                                                    class_mode='sparse', classes=class_names,
                                                    subset='training', shuffle=True)

    val_data_gen = ImageDataGenerator()
    val_data = val_data_gen.flow_from_directory(data_folder, target_size=image_size, batch_size=batch_size,
                                                class_mode='sparse', classes=class_names,
                                                subset='validation', shuffle=False)

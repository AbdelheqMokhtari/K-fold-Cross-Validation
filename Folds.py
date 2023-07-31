import os
import shutil

import cv2
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from keras.utils import to_categorical

# Uncomment the path based on the dataset you are working with.
data_path = 'Species'
# data_path = 'Varieties'
# data_path = 'Impurties'

root_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.'))  # Get the absolute path of the project root folder
data_folder = os.path.join(root_folder, 'Data', data_path)  # Folder containing the dataset

print(data_folder)

# Get the class names based on the folder names
class_names = sorted(os.listdir(data_folder))

# Load images and labels from the data folder
X_train, Y_train = [], []
train_image_paths = []
for class_index, class_name in enumerate(class_names):
    class_folder = os.path.join(data_folder, class_name)
    image_path = []
    for image_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, image_name)
        print(img_path)
        image_path.append(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))  # Resize if needed
        X_train.append(image)
        Y_train.append(class_index)

    # Calculate the split index (80%)
    split_index = int(len(image_path) * 0.8)
    image_paths.append(image_path)


Y_train = np.array(Y_train)

# Assuming you have a list of image paths called image_paths and corresponding labels called labels
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize a list to store the folds
folds = []

# Convert image_paths and labels to arrays to use in StratifiedKFold.split()
image_paths_array = np.array(image_paths)
labels_array = np.array(Y_train)
print('-----------------------------------------------------------------------------------')
# Split the data into folds while ensuring the class distribution is maintained
fold = 1
for train_index, test_index in skf.split(image_paths_array, labels_array):
    train_paths = image_paths_array[train_index]
    test_paths = image_paths_array[test_index]
    print(train_paths[0])
    print(test_paths[0])

    for image_path in train_paths:
        dst_dict = os.path.join('DataSet', f'Fold{fold}', 'train',
                                os.path.splitext(os.path.basename(image_path))[0].split('_')[0])
        if not os.path.exists(dst_dict):
            # Create the directory
            os.makedirs(dst_dict)
        dst_path = os.path.join(dst_dict, os.path.basename(image_path))
        shutil.copy(image_path, dst_path)

    for image_path in test_paths:
        dst_dict = os.path.join('DataSet', f'Fold{fold}', 'validation',
                                os.path.splitext(os.path.basename(image_path))[0].split('_')[0])
        if not os.path.exists(dst_dict):
            # Create the directory
            os.makedirs(dst_dict)
        dst_path = os.path.join(dst_dict, os.path.basename(image_path))
        shutil.copy(image_path, dst_path)

    fold += 1




















#for fold, (train_index, val_index) in enumerate(kf.split(image_paths)):
#    # Get the training and validation indices for this fold
#    train_indices = train_index[:int(len(train_index) * 0.8)]
#    val_indices = train_index[int(len(train_index) * 0.8):]

#    print(train_indices)
#    print(val_indices)

    # Load and preprocess the images for training and validation
#    train_images = [image_paths[i] for i in train_indices]
#    val_images = [image_paths[i] for i in val_indices]

#    print(train_images[0])
#    print(val_images[0])

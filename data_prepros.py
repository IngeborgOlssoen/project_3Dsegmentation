import os
import SimpleITK as sitk

import numpy as np
import torch


import SimpleITK as sitk
import numpy as np

def load_nrrd_file(file_path):
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)  # Convert to numpy array
    return arr

def custom_nrrd_reader(file_path):
    data = load_nrrd_file(file_path)
    return {'data': data, 'metadata': {}}


def get_data_pairs(data_dir):
    ctca_dir = os.path.join(data_dir, 'CTCA')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    ctca_files = sorted(os.listdir(ctca_dir))
    annotation_files = sorted(os.listdir(annotations_dir))

    data_pairs = []
    for ctca_file, annotation_file in zip(ctca_files, annotation_files):
        ctca_path = os.path.join(ctca_dir, ctca_file)
        annotation_path = os.path.join(annotations_dir, annotation_file)

        # Ensure matching filenames if needed
        if ctca_file.split('.')[0] == annotation_file.split('.')[0]:
            data_pairs.append({
                'image': ctca_path,
                'label': annotation_path
            })
        else:
            print(f"Warning: Mismatch found between {ctca_file} and {annotation_file}")

    return data_pairs

# Specify  data directories
diseased_dir = '/datasets/tdt4265/mic/asoca/Diseased'
normal_dir = '/datasets/tdt4265/mic/asoca/Normal'

# Get pairs
diseased_pairs = get_data_pairs(diseased_dir)
normal_pairs = get_data_pairs(normal_dir)

# Combined pairs for both diseased and normal data for training
combined_pairs = diseased_pairs + normal_pairs

from sklearn.model_selection import StratifiedShuffleSplit

# First, you need labels for each pair to stratify them
labels = [1 if 'diseased' in pair['image'] else 0 for pair in combined_pairs]

# Create the StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

# Perform the split
for train_index, test_val_index in sss.split(combined_pairs, labels):
    # Split off the training set
    train_pairs = [combined_pairs[i] for i in train_index]
    
    # The remaining pairs are split into validation and test sets
    test_val_pairs = [combined_pairs[i] for i in test_val_index]
    labels_test_val = [labels[i] for i in test_val_index]

# Now we further split the test_val_pairs into validation and test sets using another split
sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val_test.split(test_val_pairs, labels_test_val):
    validate_pairs = [test_val_pairs[i] for i in val_index]
    test_pairs = [test_val_pairs[i] for i in test_index]

# Output the number of pairs in each set
print(f"Number of training pairs: {len(train_pairs)}")
print(f"Number of validation pairs: {len(validate_pairs)}")
print(f"Number of test pairs: {len(test_pairs)}")



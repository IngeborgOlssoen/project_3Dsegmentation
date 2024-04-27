import os

def get_data_quad(data_dir, distance_map_dir):
    # Existing directories and files
    ctca_dir = os.path.join(data_dir, 'CTCA')
    annotations_dir = os.path.join(data_dir, 'Annotations')
    centerlines_dir = os.path.join(data_dir, 'Centerlines')
    surface_meshes_dir = os.path.join(data_dir, 'SurfaceMeshes')

    ctca_files = sorted(os.listdir(ctca_dir))
    annotation_files = sorted(os.listdir(annotations_dir))
    centerline_files = sorted(os.listdir(centerlines_dir))
    surface_mesh_files = sorted(os.listdir(surface_meshes_dir))

    # Assuming distance map files are named as 'distance_map_diseased_{index}.npy' or 'distance_map_normal_{index}.npy'
    distance_map_files = sorted(os.listdir(distance_map_dir))

    data_quad = []
    for ctca_file, annotation_file, centerline_file, surface_mesh_file, distance_map_file in zip(ctca_files, annotation_files, centerline_files, surface_mesh_files, distance_map_files):
        if ctca_file.split('.')[0] == annotation_file.split('.')[0] == centerline_file.split('.')[0] == surface_mesh_file.split('.')[0] == distance_map_file.split('.')[0].rsplit('_', 1)[0]:
            ctca_path = os.path.join(ctca_dir, ctca_file)
            annotation_path = os.path.join(annotations_dir, annotation_file)
            centerline_path = os.path.join(centerlines_dir, centerline_file)
            surface_mesh_path = os.path.join(surface_meshes_dir, surface_mesh_file)
            distance_map_path = os.path.join(distance_map_dir, distance_map_file)
            
            data_quad.append({
                'image': ctca_path,
                'label': annotation_path,
                'centerline': centerline_path,
                'surface_mesh': surface_mesh_path,
                'distance_map': distance_map_path
            })
        else:
            print(f"Warning: Mismatch found between {ctca_file}, {annotation_file}, {centerline_file}, {surface_mesh_file}, and {distance_map_file}")

    return data_quad

# Update how you call get_data_quad with the additional distance_map_dir parameter
diseased_dir = '/datasets/tdt4265/mic/asoca/Diseased'
normal_dir = '/datasets/tdt4265/mic/asoca/Normal'
distance_map_dir = '/work/ingesols/project_3Dsegmentation/distance_maps'

diseased_quads = get_data_quad(diseased_dir, distance_map_dir)
normal_quads = get_data_quad(normal_dir, distance_map_dir)


# Combined pairs for both diseased and normal data for training
combined_quad = diseased_quads + normal_quads

from sklearn.model_selection import StratifiedShuffleSplit

# First, ensure the labels are extracted from the correct dataset structure
labels = [1 if 'diseased' in pair['image'] else 0 for pair in combined_quad]

# Create the StratifiedShuffleSplit object
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

# Perform the split on the updated combined_quad
for train_index, test_val_index in sss.split(combined_quad, labels):
    # Split off the training set from the updated dataset
    train_quads = [combined_quad[i] for i in train_index]
    
    # The remaining pairs from the updated dataset are split into validation and test sets
    test_val_quads = [combined_quad[i] for i in test_val_index]
    labels_test_val = [labels[i] for i in test_val_index]

# Now we further split the test_val_pairs into validation and test sets using another split
sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_index, test_index in sss_val_test.split(test_val_quads, labels_test_val):
    validate_quads = [test_val_quads[i] for i in val_index]
    test_quads = [test_val_quads[i] for i in test_index]

# Output the number of pairs in each set to confirm the splits
print(f"Number of training pairs: {len(train_quads)}")
print(f"Number of validation pairs: {len(validate_quads)}")
print(f"Number of test pairs: {len(test_quads)}")


# Assuming combined_quad is your dataset list containing all the data entries
for entry in combined_quad:
    print("Image Path: ", entry['image'])
    print("Label Path: ", entry['label'])
    print("Surface Mesh Path: ", entry['surface_mesh'])
    print("Centerline Path: ", entry['centerline'])
    print("Distance Map Path: ", entry['distance_map'])  # Ensure this is included
    print("\n")  # Adding a newline for better readability between entries


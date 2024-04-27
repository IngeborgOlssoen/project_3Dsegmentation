import os
import glob
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

def vtp_to_binary_mask(vtp_file, image_shape, spacing, origin):
    # Read the VTK file
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_file)
    reader.Update()
    polydata = reader.GetOutput()
    
    # Extract points from polydata
    points = vtk_to_numpy(polydata.GetPoints().GetData())

    # Create an empty binary mask with the same dimensions as the CT scan
    binary_mask = np.zeros(image_shape, dtype=bool)

    # Convert world coordinates to voxel indices
    voxel_indices = np.rint((points - origin) / spacing).astype(int)
    
    # Mark the voxels corresponding to the centerline
    valid_indices = (voxel_indices >= 0).all(axis=1) & (voxel_indices < np.array(image_shape)).all(axis=1)
    voxel_indices = voxel_indices[valid_indices]
    binary_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    
    # Optionally apply dilation to the mask to widen the centerline effect
    dilated_mask = binary_dilation(binary_mask, structure=np.ones((3, 3, 3)))
    
    return dilated_mask

def process_vtp_files(base_dir, image_shape, spacing, origin):
    categories = ["Normal", "Diseased"]
    for category in categories:
        centerline_dir = os.path.join(base_dir, category, "Centerlines")
        vtp_files = glob.glob(os.path.join(centerline_dir, '*.vtp'))
        for vtp_file in vtp_files:
            print(f"Processing {vtp_file}...")
            binary_mask = vtp_to_binary_mask(vtp_file, image_shape, spacing, origin)

def visualize_centerline_overlay(ct_image, binary_mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_image, cmap='gray')
    plt.imshow(binary_mask, alpha=0.5, cmap='jet')  # Overlay mask
    plt.title('Centerline Overlay')
    plt.subplot(1, 2, 2)
    plt.imshow(ct_image, cmap='gray')
    plt.title('Original CT Image')
    plt.show()

# Placeholder functions that need implementation based on specific project requirements
def load_ct_image(ct_image_path):
    # Placeholder for loading a CT image
    return np.random.rand(512, 512, 300)  # Random data as a placeholder

def perturb_centerlines(binary_mask):
    # Placeholder for data augmentation by perturbing centerlines
    return binary_mask

def read_vtp(vtp_file):
    # Placeholder for reading VTP files and extracting geometric data
    return None

def calculate_centerline_features(centerline_data):
    # Placeholder for calculating geometric features from centerlines
    return {}

# Example Usage
base_dir = '/datasets/tdt4265/mic/asoca'
image_shape = (512, 512, 300)
spacing = (1.0, 1.0, 1.0)
origin = (0, 0, 0)

process_vtp_files(base_dir, image_shape, spacing, origin)

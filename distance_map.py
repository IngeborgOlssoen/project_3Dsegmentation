import vtk
from vtk.util.numpy_support import vtk_to_numpy
import SimpleITK as sitk
import numpy as np
import os

def compute_distance_map(image_path, mesh_path):
    # Load the image with SimpleITK
    sitk_image = sitk.ReadImage(image_path)
    numpy_image = sitk.GetArrayFromImage(sitk_image)
    
    # Create an empty vtkImageData object with the same dimensions and origin as the SimpleITK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(sitk_image.GetSize())
    vtk_image.SetSpacing(sitk_image.GetSpacing())
    vtk_image.SetOrigin(sitk_image.GetOrigin())

    # Load the STL file
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()
    polydata = reader.GetOutput()

    # Compute distance field over image volume
    distance_filter = vtk.vtkImplicitPolyDataDistance()
    distance_filter.SetInput(polydata)

    # Create an array to store the distance values
    distances = np.zeros(numpy_image.shape, dtype=float)
    for z in range(numpy_image.shape[0]):
        for y in range(numpy_image.shape[1]):
            for x in range(numpy_image.shape[2]):
                # Convert voxel coordinate to world coordinate
                world_x, world_y, world_z = vtk_image.GetOrigin()
                world_x += x * vtk_image.GetSpacing()[0]
                world_y += y * vtk_image.GetSpacing()[1]
                world_z += z * vtk_image.GetSpacing()[2]
                # Get the distance to the nearest point on the mesh
                distances[z, y, x] = distance_filter.EvaluateFunction((world_x, world_y, world_z))

    return distances

def process_all_files(stl_dir, nrrd_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    stl_files = sorted([f for f in os.listdir(stl_dir) if f.endswith('.stl')])
    nrrd_files = sorted([f for f in os.listdir(nrrd_dir) if f.endswith('.nrrd')])

    for stl_file, nrrd_file in zip(stl_files, nrrd_files):
        stl_path = os.path.join(stl_dir, stl_file)
        nrrd_path = os.path.join(nrrd_dir, nrrd_file)
        
        print(f"Processing {stl_file} and {nrrd_file}")
        distance_map = compute_distance_map(nrrd_path, stl_path)
        
        # Save the distance map
        output_path = os.path.join(output_dir, f'distance_map_{nrrd_file[:-5]}.npy')
        np.save(output_path, distance_map)
        print(f"Saved distance map to {output_path}")


# Example usage
stl_directory = '/datasets/tdt4265/mic/asoca/Diseased/SurfaceMeshes'
nrrd_directory = '/datasets/tdt4265/mic/asoca/Diseased/CTCA'
output_directory = '/work/ingesols/project_3Dsegmentation/distance_maps'

process_all_files(stl_directory, nrrd_directory, output_directory)

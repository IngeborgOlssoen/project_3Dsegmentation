import os
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from data_prepros import combined_quad

def load_nrrd_file(file_path):
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)  # Convert to numpy array
    return arr

def load_vtp_file(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    points = vtk_to_numpy(polydata.GetPoints().GetData())
    return points

def load_stl_file(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    surface = vtk_to_numpy(polydata.GetPoints().GetData())
    return surface

# Function to load data and print their shapes
def load_and_print_shapes(data_pairs):
    for entry in data_pairs:
        image_data = load_nrrd_file(entry['image'])
        label_data = load_nrrd_file(entry['label'])
        centerline_data = load_vtp_file(entry['centerline'])
        surface_mesh_data = load_stl_file(entry['surface_mesh'])
        
        print("Image Path:", entry['image'], "Shape:", image_data.shape)
        print("Label Path:", entry['label'], "Shape:", label_data.shape)
        print("Surface Mesh Path:", entry['surface_mesh'], "Shape:", surface_mesh_data.shape)
        print("Centerline Path:", entry['centerline'], "Shape:", centerline_data.shape)
        print("\n")

# Assuming combined_quad is your dataset list containing all the data entries
load_and_print_shapes(combined_quad)

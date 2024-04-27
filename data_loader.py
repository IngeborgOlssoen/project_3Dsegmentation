from monai.data import Dataset, DataLoader


from monai.data import pad_list_data_collate

from monai.transforms import Compose, LoadImaged, Resize, ToTensord, EnsureTyped, ResizeWithPadOrCropd
from monai.transforms import Compose, LoadImaged, ResizeWithPadOrCropd, ToTensord
from monai.data import Dataset, DataLoader

from monai.transforms import  EnsureChannelFirstd, CastToTyped

from data_prepros import train_pairs, validate_pairs, test_pairs
from monai.inferers.inferer import SlidingWindowInferer
import torch
import numpy as np

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, CastToTyped, ResizeWithPadOrCropd,
    ScaleIntensityRanged, Spacingd, RandCropByPosNegLabeld, RandRotate90d,
    RandFlipd, RandShiftIntensityd, ToTensord
)
from monai.transforms import Transform
import vtk
from vtk.util.numpy_support import vtk_to_numpy


train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    LoadImaged(dist),
    EnsureChannelFirstd(keys=["image", "label"]),
    CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=8, neg=1, num_samples=4),
    ToTensord(keys=["image", "label", "centerline", "surface_mesh"])
])



validate_transforms = Compose([
    LoadImaged(keys=["image", "label", "center_lines", "surfacemeshes"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # Apply channel-first only to images and labels
    CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ToTensord(keys=["image", "label", "center_lines", "surfacemeshes"])  # Convert all to tensor
])

model_inference= SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)



#Dataloader setup
# Assume 'data_pairs' is a list of dictionaries: [{'image': img_path, 'label': label_path}, ...]
train_ds = Dataset(data=train_pairs, transform=train_transforms)
validate_ds = Dataset(data=validate_pairs, transform=validate_transforms)
test_ds = Dataset(data=test_pairs, transform=train_transforms)

train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=4,pin_memory=True)
validate_loader = DataLoader(validate_ds, batch_size=1, shuffle=True,num_workers=4,pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)









##################DEBUGGING######################
# Check initial memory usage
def print_memory_usage():
    print(f"Current memory usage: {torch.cuda.memory_allocated() / 1e9} GB")

# teration counter
for i, batch_data in enumerate(train_loader):
    images, labels = batch_data["image"], batch_data["label"]
    
    
    if i % 10 == 0:  # Print every 10 batches
        
        print(f"Batch {i}: Image shape: {images.shape}, Label shape: {labels.shape}")
        print(f"Batch {i}: Unique labels: {labels.unique()}")
        print(f"Batch {i}: Labels shape: {labels.shape}, Labels dtype: {labels.dtype}")
        print_memory_usage()



def debug_print(data):
    print({k: v.shape for k, v in data.items() if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)})
    return data

def print_shape(data):
    # Assuming data is loaded and `image` and `label` are numpy arrays or tensors
    print(f"Image shape: {data['image'].shape}, Label shape: {data['label'].shape}")



class DebugTransform(Transform):
    def __call__(self, data):
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                print(f"{key}: array shape {data[key].shape}")
            elif isinstance(data[key], torch.Tensor):
                print(f"{key}: tensor shape {data[key].shape}")
            else:
                print(f"{key}: type {type(data[key])}")
        return data
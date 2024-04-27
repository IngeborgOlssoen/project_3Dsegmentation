from monai.networks.nets import UNet
from torch import optim
import torch.nn as nn

from monai.networks.nets import UNet
import torch

# Set up GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# Define your model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2), #2,2,2,2
    num_res_units=2, #2
    norm='INSTANCE'  # Assume you have batch normalization as per your earlier setup
)

# Move model to GPU
model.to(device)

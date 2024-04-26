import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from monai.metrics import compute_hausdorff_distance
from data_loader import validate_loader  # Assuming you have a validation loader similar to train_loader
from model import model
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer


# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate_model(model, data_loader, num_visualizations=3):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients needed for evaluation, which saves memory and computations
        dice_metric =DiceMetric(include_background=False, reduction="mean")
        hd95_scores = []
        dice_scores = []
        count = 0
        
        for batch_data in data_loader:
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            #outputs = model(images)
            inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)
            outputs = inferer(inputs=images, network=model)

            print(f"Output shape: {outputs.shape}, Label shape: {labels.shape}")
            
            # Compute Dice score
            dice_metric(y_pred=outputs, y=labels)
            #dice_scores.append(dice_score)

            # Compute HD95
            outputs_bin = (outputs > 0.1)  # Lowering threshold to see if it captures more of the foreground class

            hd95_score = compute_hausdorff_distance(y_pred=outputs_bin, y=labels, include_background=False, percentile=95)
            hd95_scores.append(hd95_score.cpu().numpy())
            

            # Visualization logic
            if len(dice_scores) < num_visualizations:
                image = images[0].cpu().squeeze()  # Assuming batch size of 1 for simplicity
                output_bin = outputs_bin[0].cpu().squeeze()
                label = labels[0].cpu().squeeze()
                fig, axs = plt.subplots(3, 3, figsize=(15, 15))
#
                # Display different slices
                slice_indices = [image.shape[0] // 2, image.shape[1] // 2, image.shape[2] // 2]
                titles = ['Axial View', 'Coronal View', 'Sagittal View']
                for i, slice_index in enumerate(slice_indices):
                    axs[0, i].imshow(image[slice_index, :, :], cmap='gray')
                    axs[0, i].set_title(f'{titles[i]} - Input')
                    axs[0, i].axis('off')
#
                    axs[1, i].imshow(label[slice_index, :, :], cmap='gray')
                    axs[1, i].set_title(f'{titles[i]} - True Mask')
                    axs[1, i].axis('off')
#
                    axs[2, i].imshow(output_bin[slice_index, :, :], cmap='gray')
                    axs[2, i].set_title(f'{titles[i]} - Predicted Mask')
                    axs[2, i].axis('off')
#
                plt.show()
                count += 1

        # Compute average metrics over all batches
        #avg_dice_score = np.mean(dice_scores)
        avg_hd95_score = np.mean(hd95_scores)
        avg_dice_score = dice_metric.aggregate().item()  # This converts tensor to scalar
        dice_metric.reset()  # Reset the metric for future use

        return avg_dice_score, avg_hd95_score


        return dice_score, avg_hd95_score

# Evaluation
avg_dice, avg_hd95 = evaluate_model(model, validate_loader)  # Use your validation dataset loader

logging.info(f"Evaluation - Average Dice Score: {avg_dice}, Average HD95: {avg_hd95}")


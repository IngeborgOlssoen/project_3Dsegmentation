import logging
from model import model
from data_loader import train_loader,validate_loader
from losses import CombinedLoss
from torch import optim
import torch
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

epochs = 50
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# Define scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
loss_func = DiceCELoss(to_onehot_y=True, softmax=True)
early_stopping_patience=10
best_val_loss=float('inf')
patience_counter=0
accumulation_steps = 4  # Set this to the number of steps over which to accumulate gradients
optimizer.zero_grad()
model_path = 'best_model.pth'
train_losses = []
val_losses = []
best_model = None 

def validate(model, loader, loss_func):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_data in loader:
            images, labels = batch_data["image"], batch_data["label"]
            images = images.to(device)
            labels = labels.to(device)
            inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)
            outputs = inferer(inputs=images, network=model)
            
            loss = loss_func(outputs, labels)
            total_val_loss += loss.item()

    return total_val_loss / len(loader)


for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    num_batches = len(train_loader)
    
    for i, batch_data in enumerate(train_loader):
        images, labels = batch_data["image"], batch_data["label"]
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        
        outputs = model(images)
        
        loss = loss_func(outputs, labels)/accumulation_steps

        # Backward pass and optimize
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # Update parameters only after accumulation steps
            optimizer.zero_grad()  # Clear gradients after updating

        epoch_loss += loss.item()*accumulation_steps  # Correct loss scaling

        #loss = loss_func(outputs, labels)

        # Backward pass and optimize
        #loss.backward()
        #optimizer.step()  # Update parameters only after accumulation steps
        #optimizer.zero_grad()  # Clear gradients after updating

        epoch_loss += loss.item()  # Correct loss scaling

        
        # Logging every 10 batches
        if i % 10 == 0:
            logging.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}, Batch Loss: {loss.item()}")

    # Validation loss calculation after the epoch
    val_loss = validate(model, validate_loader, loss_func)
    val_losses.append(val_loss)
    train_losses.append(epoch_loss / len(train_loader))
    logging.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

    # Save the best model and implement early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()  # Save the model state
        torch.save(model.state_dict(), model_path)
        logging.info(f"Saved improved model to {model_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            model.load_state_dict(best_model)  # Revert to best model
            torch.save(model.state_dict(), 'final_model.pth')
            logging.info(f"Early stopping triggered. Best model saved to 'final_model.pth'")
            break

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_curves.png')

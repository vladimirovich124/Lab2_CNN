import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn
    (outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn
        (outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc

def early_stopping(val_loss, best_val_loss, patience, counter):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
    return best_val_loss, counter

def train_model(model, train_loader, val_loader, num_epochs, lr, device, logger, patience=5, save_dir=None):
    loss_fn
 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        train_loss = train_epoch(model, train_loader, loss_fn
    , optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, loss_fn
    , device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        best_val_loss, early_stop_counter = early_stopping(val_loss, best_val_loss, patience, early_stop_counter)

        if early_stop_counter >= patience:
            logger.info("Early stopping triggered. Training stopped.")
            break

    plot_training_curves(train_losses, val_losses, val_accuracies, save_dir)

    return train_losses, val_losses, val_accuracies

def plot_training_curves(train_losses, val_losses, val_accuracies, save_dir=None):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'training_validation_loss.png'))
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'validation_accuracy.png'))
    plt.show()
import torch.nn.functional as F
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import numpy as np
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
from .data.dataset import ClassificationDataset
from .model.classification.classification_model import *
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from .data.utils import split_indices


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 26
learning_rate = 1e-3
batch_size = 128
num_epochs = 30
input_size = 11
a, b = 0.5, 1

# Load Data
dataset = ClassificationDataset(
    csv_file="power.csv", root_dir="test123", transform=transforms.ToTensor()
)
train_set, test_set = torch.utils.data.random_split(dataset, [2900, 57])

val_pct = 0.3
rand_seed = 42
train_indices, val_indices = split_indices(len(dataset), val_pct, rand_seed)


# Training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)

# Validation set and data loader
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = ClassificationModel(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


print(len(train_set))
print(len(test_set))


# Train Network
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    # Generate predictions
    local_pred, global_pred, fusion_pred = model(xb)
    # Calculate loss
    # print(max(yb))
    loss_lc = loss_func(local_pred, yb)
    loss_gb = loss_func(global_pred, yb)
    loss_fs = loss_func(fusion_pred, yb)
    loss = a * (loss_lc + loss_gb) + b * loss_fs

    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()
    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(fusion_pred, yb)
    return (
        loss.item(),
        loss_lc.item(),
        loss_gb.item(),
        loss_fs.item(),
        len(xb),
        metric_result,
    )


def evaluate(model, loss_func, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [
            loss_batch(model, loss_func, xb, yb, metric=metric) for xb, yb in valid_dl
        ]
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the data set
        total = np.sum(nums)
        # Avg, loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def fit(
    epochs, model, loss_func, train_dl, valid_dl, opt_fn=None, lr=None, metric=None
):
    train_losses, val_losses, val_metrics = [], [], []
    torch.cuda.empty_cache()
    # Instantiate the optimizer
    if opt_fn is None:
        opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, mode="min", patience=8, min_lr=1e-4, verbose=True
    )

    for epoch in range(epochs):
        # Training
        model.train()
        for xb, yb in train_dl:
            train_loss, loss_lc, loss_gb, loss_fs, _, _ = loss_batch(
                model, loss_func, xb, yb, opt
            )

        # Evaluation
        model.eval()
        result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric)
        val_loss, total, val_metric = result
        sched.step(val_loss)
        # Record the loss and metric
        all_loss = [train_loss, loss_lc, loss_gb, loss_fs]
        train_losses.append(all_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        # Print progress
        if metric is None:
            print(
                "Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}".format(
                    epoch + 1, epochs, train_loss, val_loss
                )
            )
        else:
            print(
                "Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}, val_{}: {:4f}".format(
                    epoch + 1, epochs, train_loss, val_loss, metric.__name__, val_metric
                )
            )
    return train_losses, val_losses, val_metrics

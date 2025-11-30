# A4codes.py
# Do NOT include top-level code (no training runs, no prints).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


############################################
# Data Loading Utilities
############################################

def _make_loaders(in_path, out_path, batch_size=32):
    """Create dataloaders for in-domain (labeled) and out-domain (unlabeled) data."""

    # In-domain: slight augmentation for robustness
    in_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Out-domain: no augmentation needed
    out_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    in_dataset = datasets.ImageFolder(in_path, transform=in_transform)
    out_dataset = datasets.ImageFolder(out_path, transform=out_transform)

    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(in_dataset.classes)

    return in_loader, out_loader, num_classes


############################################
# Entropy Loss (used to worsen out-domain classification)
############################################

def _entropy_loss(logits):
    """Encourage high entropy (uniform predictions) for out-domain samples."""
    probs = torch.softmax(logits, dim=1)
    logp = torch.log(torch.clamp(probs, 1e-12, 1.0))
    entropy = -torch.sum(probs * logp, dim=1)
    return entropy.mean()


############################################
# QUESTION 1: learn()
############################################

def learn(in_domain_path, out_domain_path):
    """
    Train a classifier using in-domain labeled data.
    Also uses out-domain unlabeled data to reduce generalization
    (via entropy maximization).
    Returns a PyTorch model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    in_loader, out_loader, num_classes = _make_loaders(
        in_domain_path, out_domain_path
    )

    ############################################
    # Model setup
    ############################################
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze entire backbone → only train final layer
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)

    ############################################
    # Optimizer & loss
    ############################################
    optimizer = optim.Adam(model.fc.parameters(), lr=3e-4)
    ce_loss = nn.CrossEntropyLoss()
    lambda_entropy = 6.0  # strong entropy push to reduce out-domain accuracy

    # out-domain iterator
    out_iter = iter(out_loader)

    model.train()
    epochs = 10

    for _ in range(epochs):
        for x_in, y_in in in_loader:

            x_in = x_in.to(device)
            y_in = y_in.to(device)

            # Get out-domain batch (cycle when exhausted)
            try:
                x_out, _ = next(out_iter)
            except StopIteration:
                out_iter = iter(out_loader)
                x_out, _ = next(out_iter)
            x_out = x_out.to(device)

            # Forward passes
            logits_in = model(x_in)
            logits_out = model(x_out)

            loss_in = ce_loss(logits_in, y_in)
            loss_out = _entropy_loss(logits_out)

            # Total loss
            loss = loss_in + lambda_entropy * loss_out

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


############################################
# QUESTION 1: compute_accuracy()
############################################

def compute_accuracy(eval_path, model):
    """
    Compute accuracy on a labeled evaluation folder.
    Works for both in-domain and out-domain datasets.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = datasets.ImageFolder(eval_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

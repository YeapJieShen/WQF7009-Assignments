"""Fine-tune, evaluate, save and load the VGG16 brain tumour classifier."""

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from brain_mri_xai import SEED
from brain_mri_xai.data import CLASSES, load_images, normalise

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
WEIGHTS_PATH = MODEL_DIR / "vgg16_brain_mri.pt"


def device() -> torch.device:
    """Return the first CUDA GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN pick deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_vgg16(n_classes: int = len(CLASSES), pretrained: bool = True) -> nn.Module:
    """VGG16 (Simonyan & Zisserman, 2015) with its last layer replaced for ``n_classes``.

    Args:
        n_classes: Number of output classes.
        pretrained: Start from the ImageNet weights (transfer learning), as in ``05_lrp.py``.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)
    return model


def imagenet_features(paths: pd.Series, batch_size: int = 64) -> np.ndarray:
    """Describe each image by the ImageNet-trained VGG16 convolutional features (not fine-tuned).

    The 512 feature maps of the last convolutional block are averaged over the image and scaled
    to unit length.

    Args:
        paths: Image files.
        batch_size: Images per forward pass.

    Returns:
        ``float32`` array (n, 512).
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device()).eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = normalise(load_images(paths.iloc[start : start + batch_size]), per_image=False).to(
                device()
            )
            features = extractor(images).mean(dim=(2, 3))
            batches.append(nn.functional.normalize(features, dim=1).cpu().numpy())
    return np.concatenate(batches)


def _run_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimiser: torch.optim.Optimizer | None
) -> tuple[float, float]:
    """One pass over ``loader``; trains if an optimiser is given. Returns (mean loss, accuracy)."""
    model.train(optimiser is not None)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(optimiser is not None):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device()), targets.to(device())
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if optimiser is not None:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(targets)
            correct += (outputs.argmax(1) == targets).sum().item()
            n += len(targets)
    return total_loss / n, correct / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int = 15,
    learning_rate: float = 1e-5,
) -> tuple[nn.Module, pd.DataFrame]:
    """Fine-tune all layers with Adam and cross-entropy; keep the epoch with the lowest validation loss.

    Args:
        model: Network on :func:`device`.
        train_loader: Augmented training batches.
        validation_loader: Validation batches, used only for model selection.
        epochs: Passes over the training data.
        learning_rate: Adam learning rate (``05_lrp.py`` uses 1e-5).

    Returns:
        The model with the best weights loaded, and a per-epoch history
        (``epoch``, ``train_loss``, ``train_accuracy``, ``validation_loss``, ``validation_accuracy``).
    """
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, loss_fn, optimiser)
        validation_loss, validation_accuracy = _run_epoch(model, validation_loader, loss_fn, None)
        history.append((epoch, train_loss, train_accuracy, validation_loss, validation_accuracy))
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    columns = ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"]
    return model, pd.DataFrame(history, columns=columns)


@torch.no_grad()
def predict_proba(
    model: nn.Module, images: torch.Tensor, batch_size: int = 64, per_image: bool = True
) -> np.ndarray:
    """Return softmax class probabilities, shape (n, classes), for ``uint8`` images.

    ``per_image`` is passed to :func:`~brain_mri_xai.data.normalise`.
    """
    model.eval()
    probabilities = [
        torch.softmax(model(normalise(batch, per_image).to(device())), dim=1).cpu()
        for batch in images.split(batch_size)
    ]
    return torch.cat(probabilities).numpy()


def save_weights(model: nn.Module, path: Path = WEIGHTS_PATH) -> None:
    """Save the model's ``state_dict`` (``models/`` is gitignored)."""
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path = WEIGHTS_PATH) -> nn.Module:
    """Rebuild the VGG16 classifier with the saved weights, in evaluation mode on :func:`device`."""
    model = build_vgg16(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device()).eval()


def randomise_top_layers(model: nn.Module, seed: int = SEED) -> nn.Module:
    """Re-initialise the last convolutional block (conv5) and the dense layers, in place.

    Used for the model-randomisation sanity check (Adebayo et al., 2018): an explanation that
    reflects what the model learned should change when these weights are replaced.
    """
    torch.manual_seed(seed)
    for layer in [*model.features[24:], *model.classifier]:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return model


def median_confidence_case(predictions: pd.DataFrame) -> int:
    """Index of the row with the (lower) median ``confidence``: a typical rather than a best case."""
    ordered = predictions.sort_values("confidence", kind="stable")
    return ordered.index[(len(ordered) - 1) // 2]

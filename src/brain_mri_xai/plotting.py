"""Image and heatmap panels for the MRI notebooks (theme and saving come from ``stroke_shap.plotting``)."""

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from brain_mri_xai.data import denormalise

CLASS_COLORS = {
    "glioma_tumor": "#2a78d6",
    "meningioma_tumor": "#eb6834",
    "no_tumor": "#b4b2a9",
    "pituitary_tumor": "#1baf7a",
}
"""Blue / orange / green from the shared palette for the tumours, grey for no tumour."""


def lrp_heatmap(relevance: torch.Tensor) -> np.ndarray:
    """Sum LRP relevance over the colour channels, shape (H, W)."""
    return relevance.sum(dim=0).numpy()


def show_image(ax: Axes, image: torch.Tensor) -> AxesImage:
    """Draw one normalised image in grey scale without axes."""
    ax.set_axis_off()
    return ax.imshow(denormalise(image).mean(dim=0), cmap="gray", vmin=0, vmax=1)


def show_lrp(ax: Axes, heatmap: np.ndarray, clip: float = 99.5) -> AxesImage:
    """Draw an LRP heatmap: red = evidence for the class, blue = against, white = none.

    The colour scale is symmetric around zero and clipped at the ``clip`` percentile of
    absolute relevance, so a few extreme pixels do not wash out the rest.
    """
    ax.set_axis_off()
    limit = np.percentile(np.abs(heatmap), clip)
    return ax.imshow(heatmap, cmap="seismic", vmin=-limit, vmax=limit)


def show_grad_cam(ax: Axes, image: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> AxesImage:
    """Draw a Grad-CAM heatmap over the grey-scale image (yellow = strongest evidence)."""
    show_image(ax, image)
    return ax.imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=alpha)

"""Layer-wise Relevance Propagation and Grad-CAM for the VGG16 classifier.

The LRP code follows ``05_lrp.py`` (deepfindr/xai-series), which is based on the LRP
tutorial of Montavon et al. (2019): LRP-gamma in the lower layers, LRP-epsilon in the
middle layers and LRP-0 in the top layers. Added here: the z^B rule for the input pixels
(the "potential step 5" left open in ``05_lrp.py``), a selectable target class, and no
dependency on the training script's globals.
"""

import copy
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from brain_mri_xai.data import IMAGENET_MEAN, IMAGENET_STD

GRADCAM_LAYER = 29
"""Index in ``model.features`` of the ReLU after the last convolution (conv5_3), 14 x 14 maps."""


def _transform(layer: nn.Module, rho: Callable[[torch.Tensor], torch.Tensor]) -> nn.Module:
    """Copy ``layer`` with ``rho`` applied to its weight and bias."""
    layer = copy.deepcopy(layer)
    layer.weight = nn.Parameter(rho(layer.weight))
    layer.bias = nn.Parameter(rho(layer.bias))
    return layer


def _dense_to_conv(classifier: nn.Sequential) -> list[nn.Module]:
    """Rewrite VGG16's dense layers as convolutions so relevance keeps its spatial shape.

    The first dense layer reads the 512 x 7 x 7 feature map, so it becomes a 7 x 7 convolution;
    the others become 1 x 1 convolutions. ReLU and dropout layers are kept as they are.
    """
    layers: list[nn.Module] = []
    for i, layer in enumerate(classifier):
        if isinstance(layer, nn.Linear):
            kernel = 7 if i == 0 else 1
            out_features, in_features = layer.weight.shape
            conv = nn.Conv2d(in_features // kernel**2, out_features, kernel)
            conv.weight = nn.Parameter(layer.weight.reshape(out_features, -1, kernel, kernel))
            conv.bias = nn.Parameter(layer.bias)
            layers.append(conv)
        else:
            layers.append(layer)
    return layers


def lrp(model: nn.Module, image: torch.Tensor, target: int | None = None) -> torch.Tensor:
    """Pixel relevance of one image for one class, by Layer-wise Relevance Propagation.

    The class logit is propagated back layer by layer; each neuron passes relevance to its
    inputs in proportion to their contribution ``a_j * w_jk``. Rules by layer index:
    0 = z^B (pixels bounded by the normalised black and white values), 1-16 = LRP-gamma
    (gamma = 0.25, favours positive evidence), 17-30 = LRP-epsilon (epsilon = 0.25 x the
    layer's root-mean-square pre-activation, keeps only salient paths), 31+ = LRP-0.
    Max pooling is treated as average pooling in the backward pass.

    Args:
        model: VGG16 from :func:`brain_mri_xai.model.build_vgg16`, in evaluation mode.
        image: One normalised image, shape (3, 224, 224).
        target: Class to explain; the predicted class if ``None``.

    Returns:
        Relevance per input value, shape (3, 224, 224), on the CPU. Positive = evidence
        for ``target``, negative = evidence against it. Sum over channels for a heatmap.
    """
    dev = next(model.parameters()).device
    layers = list(model.features) + _dense_to_conv(model.classifier)
    activations = [image.unsqueeze(0).to(dev)]
    with torch.no_grad():
        for layer in layers:
            activations.append(layer(activations[-1]))
    logits = activations[-1].flatten()
    target = int(logits.argmax()) if target is None else target

    relevance = torch.zeros_like(activations[-1])
    relevance[0, target] = activations[-1][0, target]
    for i in range(len(layers) - 1, 0, -1):
        layer = layers[i]
        if isinstance(layer, nn.MaxPool2d):
            layer = nn.AvgPool2d(2)
        if not isinstance(layer, nn.Conv2d | nn.AvgPool2d):
            continue  # ReLU and dropout pass relevance through unchanged
        a = activations[i].detach().requires_grad_(True)
        if isinstance(layer, nn.AvgPool2d):
            z = layer(a) + 1e-9
        elif i <= 16:
            z = _transform(layer, lambda p: p + 0.25 * p.clamp(min=0))(a) + 1e-9
        elif i <= 30:
            z = layer(a)
            z = z + 1e-9 + 0.25 * (z**2).mean().sqrt().detach()
        else:
            z = layer(a) + 1e-9
        (z * (relevance / z).detach()).sum().backward()
        relevance = (a * a.grad).detach()

    # z^B rule for the first convolution: pixel values are bounded by the normalised black and white
    x = activations[0].detach().requires_grad_(True)
    low = (torch.zeros_like(x) - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
    high = (torch.ones_like(x) - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
    low.requires_grad_(True)
    high.requires_grad_(True)
    first = layers[0]
    z = first(x) - _transform(first, lambda p: p.clamp(min=0))(low)
    z = z - _transform(first, lambda p: p.clamp(max=0))(high)
    # z^B is non-negative in exact arithmetic; at pure black or white pixels it can round to 0
    (z * (relevance / z.clamp(min=1e-9)).detach()).sum().backward()
    relevance = x * x.grad + low * low.grad + high * high.grad
    return relevance[0].detach().cpu()


def grad_cam(
    model: nn.Module, image: torch.Tensor, target: int | None = None, layer_index: int = GRADCAM_LAYER
) -> np.ndarray:
    """Class activation map of one image by Grad-CAM (Selvaraju et al., 2017).

    Each feature map of the chosen layer is weighted by the mean gradient of the class logit
    over that map; the weighted sum is passed through a ReLU (keeps evidence for the class)
    and upsampled to the image size.

    Args:
        model: VGG16 from :func:`brain_mri_xai.model.build_vgg16`, in evaluation mode.
        image: One normalised image, shape (3, 224, 224).
        target: Class to explain; the predicted class if ``None``.
        layer_index: Layer in ``model.features`` whose output is used.

    Returns:
        Heatmap, shape (224, 224), scaled to [0, 1] (all zeros if there is no positive evidence).
    """
    dev = next(model.parameters()).device
    stored: dict[str, torch.Tensor] = {}

    def keep_output(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Forward hook: store the layer output and keep its gradient."""
        output.retain_grad()
        stored["maps"] = output

    hook = model.features[layer_index].register_forward_hook(keep_output)
    try:
        x = image.unsqueeze(0).to(dev).requires_grad_(True)  # VGG's in-place ReLUs need a grad input
        logits = model(x)
        target = int(logits.argmax()) if target is None else target
        model.zero_grad()
        logits[0, target].backward()
    finally:
        hook.remove()
    maps, gradients = stored["maps"].detach(), stored["maps"].grad
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = functional.relu((weights * maps).sum(dim=1, keepdim=True))
    cam = functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    peak = cam.max()
    return (cam / peak if peak > 0 else cam).cpu().numpy()


def block_sums(heatmap: np.ndarray, size: int = 16) -> np.ndarray:
    """Sum a square heatmap over ``size`` x ``size`` blocks (16 = one conv5 cell), flattened.

    Comparing heatmaps block by block ignores pixel-level texture, so maps with different
    resolutions (LRP, Grad-CAM, Captum's LRP) can be rank-correlated fairly.
    """
    n = heatmap.shape[0] // size
    return heatmap.reshape(n, size, n, size).sum(axis=(1, 3)).ravel()

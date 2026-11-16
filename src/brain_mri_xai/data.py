"""Download, verify, index, de-duplicate (including near-copies), split and load the MRI images."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

from brain_mri_xai import SEED

MRI_DIR = Path(__file__).resolve().parents[2] / "data" / "brain_mri"

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
"""Folder names in ``ImageFolder`` (alphabetical) order, so list position = class index."""

CLASS_LABELS = {
    "glioma_tumor": "Glioma",
    "meningioma_tumor": "Meningioma",
    "no_tumor": "No Tumour",
    "pituitary_tumor": "Pituitary",
}
"""Display labels for figures and tables."""

IMAGE_SIZE = 224
"""Input height and width expected by the ImageNet-pretrained VGG16."""

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DATASET_REPO = "SartajBhuvaji/Brain-Tumor-Classification-DataSet"
DATASET_COMMIT = "3ea9f4eae54c67e5bddbf6f7a73582967a5f5f79"
"""The repository's only commit ("First Commit", 24 May 2020), pinned so the files cannot change."""

DATASET_URL = f"https://codeload.github.com/{DATASET_REPO}/zip/{DATASET_COMMIT}"
ARCHIVE_PATH = MRI_DIR.parent / "brain_mri_github.zip"

MANUAL_LABELS_PATH = MRI_DIR.parent / "brain_mri_manual_labels.csv"
"""Visual labels by the author (not a radiologist): modality (CT / MRI / Unclear) and MRI sequence."""


def download_archive(path: Path = ARCHIVE_PATH, url: str = DATASET_URL) -> Path:
    """Download the dataset's GitHub archive at the pinned commit, unless it is already there.

    Args:
        path: Where to keep the zip file (``data/`` is gitignored).
        url: Archive URL; defaults to the pinned commit.

    Returns:
        ``path``.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(path)
    return path


def _archive_images(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Map ``training/<class>/<file>`` paths to the archive members holding the images."""
    images = {}
    for info in archive.infolist():
        parts = info.filename.split("/")  # <repo>-<commit>/Training/<class>/<file>
        if not info.is_dir() and len(parts) == 4 and parts[1] in ("Training", "Testing"):
            images[f"{parts[1].lower()}/{parts[2]}/{parts[3]}"] = info
    return images


def compare_with_archive(archive_path: Path = ARCHIVE_PATH, root: Path = MRI_DIR) -> pd.DataFrame:
    """Compare every local image with its counterpart in the GitHub archive by SHA-256.

    Args:
        archive_path: Output of :func:`download_archive`.
        root: Local folder holding ``training/`` and ``testing/``.

    Returns:
        One row per file found in either place: ``path`` (relative), ``split``, ``in_archive``,
        ``in_local`` and ``identical`` (both present with the same SHA-256).
    """
    with zipfile.ZipFile(archive_path) as archive:
        remote = {
            path: hashlib.sha256(archive.read(info)).hexdigest()
            for path, info in _archive_images(archive).items()
        }
    local = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.glob("*/*/*")
        if path.is_file()
    }
    rows = [
        (path, path.split("/")[0], path in remote, path in local, remote.get(path) == local.get(path))
        for path in sorted(remote.keys() | local.keys())
    ]
    return pd.DataFrame(rows, columns=["path", "split", "in_archive", "in_local", "identical"])


def extract_archive(archive_path: Path = ARCHIVE_PATH, root: Path = MRI_DIR) -> None:
    """Write the archive's images to ``root/training`` and ``root/testing``, replacing local copies.

    Args:
        archive_path: Output of :func:`download_archive`.
        root: Destination folder.
    """
    with zipfile.ZipFile(archive_path) as archive:
        for path, info in _archive_images(archive).items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))


def index_images(root: Path = MRI_DIR) -> pd.DataFrame:
    """List every image with its split, class, size and content hash.

    Args:
        root: Folder holding ``training/`` and ``testing/``, each with one sub-folder per class.

    Returns:
        One row per file: ``path``, ``split`` (training/testing), ``label`` (folder name),
        ``target`` (class index), ``width``, ``height`` and ``md5`` of the file bytes.
    """
    rows = []
    for split in ["training", "testing"]:
        for target, label in enumerate(CLASSES):
            for path in sorted((root / split / label).iterdir()):
                with Image.open(path) as image:
                    width, height = image.size
                md5 = hashlib.md5(path.read_bytes()).hexdigest()
                rows.append((path, split, label, target, width, height, md5))
    return pd.DataFrame(rows, columns=["path", "split", "label", "target", "width", "height", "md5"])


NEAR_IDENTICAL_CORRELATION = 0.97
"""Pixel correlation (256 x 256) from which two images count as near-identical: the same slice saved again.

Set by inspecting every candidate pair between 0.95 and 0.995 (notebook 04)."""

RELATED_CORRELATION = 0.90
"""Pixel correlation from which two images count as related: near-identical or, mostly, a
neighbouring slice of the same scan series."""


def grey_images(paths: pd.Series, size: int) -> np.ndarray:
    """Read images in grey scale, resized to ``size`` x ``size``; ``uint8`` array (n, size, size)."""
    return np.stack(
        [
            np.asarray(Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR))
            for path in paths
        ]
    )


def _difference_hash(images: np.ndarray) -> np.ndarray:
    """64-bit hash per image: is each pixel of a 9 x 8 thumbnail brighter than its left neighbour?"""
    thumbnails = np.stack(
        [
            np.asarray(Image.fromarray(image).resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
            for image in images
        ]
    )
    bits = (thumbnails[:, :, 1:] > thumbnails[:, :, :-1]).reshape(len(images), -1)
    return np.packbits(bits, axis=1).view(">u8").ravel()


def standardise(images: np.ndarray) -> np.ndarray:
    """Flatten each image to zero mean and unit length, so the dot product of two is their correlation.

    Args:
        images: Array (n, height, width).

    Returns:
        ``float32`` array (n, height * width).
    """
    vectors = images.reshape(len(images), -1).astype(np.float32)
    vectors -= vectors.mean(axis=1, keepdims=True)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6)


def find_similar_pairs(
    index: pd.DataFrame, max_hash_distance: int = 12, min_coarse_correlation: float = 0.85
) -> pd.DataFrame:
    """Find pairs of images that may show the same or a neighbouring slice, also when one is mirrored.

    Three steps: shortlist pairs whose difference hashes differ in at most ``max_hash_distance``
    of 64 bits, keep those with a pixel correlation of at least ``min_coarse_correlation`` at
    128 x 128, and measure the final correlation at 256 x 256.

    Args:
        index: Output of :func:`index_images`.
        max_hash_distance: Hash shortlist threshold, in bits.
        min_coarse_correlation: Correlation threshold at 128 x 128.

    Returns:
        One row per pair: ``a`` < ``b`` (row positions in ``index``), ``mirrored`` (``a`` was
        flipped left-right to match ``b``) and ``correlation`` at 256 x 256.
    """
    small = grey_images(index["path"], 128)
    mirrored_small = small[:, :, ::-1]
    hashes, mirrored_hashes = _difference_hash(small), _difference_hash(mirrored_small)
    vectors, mirrored_vectors = standardise(small), standardise(mirrored_small)
    rows = []
    for a in range(len(small) - 1):
        for mirrored, image_hash, vector in [
            (False, hashes[a], vectors[a]),
            (True, mirrored_hashes[a], mirrored_vectors[a]),
        ]:
            b = a + 1 + np.flatnonzero(np.bitwise_count(image_hash ^ hashes[a + 1 :]) <= max_hash_distance)
            b = b[vectors[b] @ vector >= min_coarse_correlation]
            rows += [(a, other, mirrored) for other in b]
    pairs = pd.DataFrame(rows, columns=["a", "b", "mirrored"])

    used = np.unique(pairs[["a", "b"]])
    large = dict(zip(used, standardise(grey_images(index["path"].iloc[used], 256)), strict=True))
    large_mirrored = dict(
        zip(used, standardise(grey_images(index["path"].iloc[used], 256)[:, :, ::-1]), strict=True)
    )
    pairs["correlation"] = [
        float((large_mirrored[a] if mirrored else large[a]) @ large[b])
        for a, b, mirrored in pairs[["a", "b", "mirrored"]].itertuples(index=False)
    ]
    # keep the better of the plain and the mirrored match for each pair
    pairs = pairs.sort_values("correlation", ascending=False).drop_duplicates(["a", "b"])
    return pairs.sort_values(["a", "b"]).reset_index(drop=True)


def group_images(n: int, pairs: pd.DataFrame, threshold: float) -> np.ndarray:
    """Group images linked, directly or through other images, by pairs at or above ``threshold``.

    Returns:
        Group number per image (row position), length ``n``.
    """
    linked = pairs[pairs["correlation"] >= threshold]
    graph = coo_matrix((np.ones(len(linked)), (linked["a"], linked["b"])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def flag_duplicates(index: pd.DataFrame, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mark training images that would leak test images or repeat other training images.

    Exact copies (same MD5) always count as near-identical. A training image is dropped if it is
    related to any test image (``leak``) or if it is near-identical to an earlier kept training
    image (``near_identical``). Test images are all kept, so results stay comparable with other work on this
    dataset.

    Args:
        index: Output of :func:`index_images`.
        pairs: Output of :func:`find_similar_pairs`; computed if not given.

    Returns:
        ``index`` (fresh row index) with ``near_identical_group`` and ``related_group`` numbers and boolean
        columns ``leak``, ``near_identical`` and ``keep``.
    """
    index = index.reset_index(drop=True)
    if pairs is None:
        pairs = find_similar_pairs(index)
    first_file = index.reset_index().groupby("md5")["index"].transform("first")
    repeated_file = first_file != index.index
    file_pairs = pd.DataFrame(
        {"a": first_file[repeated_file], "b": index.index[repeated_file], "correlation": 1.0}
    )
    pairs = pd.concat([pairs, file_pairs], ignore_index=True)
    index["near_identical_group"] = group_images(len(index), pairs, NEAR_IDENTICAL_CORRELATION)
    index["related_group"] = group_images(len(index), pairs, RELATED_CORRELATION)

    is_train = index["split"] == "training"
    index["leak"] = is_train & index["related_group"].isin(index.loc[~is_train, "related_group"])
    remaining = is_train & ~index["leak"]
    index["near_identical"] = remaining & index["near_identical_group"].where(remaining).duplicated()
    index["keep"] = ~(index["leak"] | index["near_identical"])
    return index


def split_train_validation(
    index: pd.DataFrame, validation_size: float = 0.15, seed: int = SEED
) -> pd.DataFrame:
    """Add a ``subset`` column: train / validation / test.

    Related training images (same ``related_group``) stay in the same subset, so neighbouring
    slices of one scan cannot sit on both sides; classes are stratified as far as the groups allow.

    Args:
        index: Output of :func:`flag_duplicates`; only rows with ``keep`` are used.
        validation_size: Approximate share of the kept training images held out for model selection.
        seed: Seed for the grouped, stratified split.

    Returns:
        Kept rows with ``subset`` set, in the original order.
    """
    kept = index[index["keep"]].copy()
    train = kept[kept["split"] == "training"]
    folds = StratifiedGroupKFold(n_splits=round(1 / validation_size), shuffle=True, random_state=seed)
    _, validation = next(folds.split(train, train["target"], groups=train["related_group"]))
    kept["subset"] = np.where(kept["split"] == "testing", "test", "train")
    kept.loc[train.index[validation], "subset"] = "validation"
    return kept


def annotate_test(
    index: pd.DataFrame, subsets: pd.DataFrame, labels_path: Path = MANUAL_LABELS_PATH
) -> pd.DataFrame:
    """Return the test rows with their modality and whether they belong to the clean test set.

    Args:
        index: Output of :func:`flag_duplicates` (all images, used to find groups with mixed labels).
        subsets: Output of :func:`split_train_validation`.
        labels_path: Manual labels CSV.

    Returns:
        Test rows (fresh row index) with ``modality`` (CT / MRI / Unclear from the manual labels; images
        not reviewed count as MRI) and ``clean``: ``True`` for the first image of each near-identical
        group, ``False`` for its repeats and for every image of a group with conflicting class labels.
    """
    test = subsets[subsets["subset"] == "test"].reset_index(drop=True)
    labels = pd.read_csv(labels_path)
    labels = labels[labels["split"] == "testing"].set_index(["label", "file"])["modality"]
    test["modality"] = [
        labels.get((label, path.name), "MRI") for label, path in zip(test["label"], test["path"], strict=True)
    ]
    mixed = index.groupby("near_identical_group")["label"].nunique().loc[lambda n: n > 1].index
    test["clean"] = ~test["near_identical_group"].isin(mixed) & ~test["near_identical_group"].duplicated()
    return test


def load_images(paths: pd.Series, size: int = IMAGE_SIZE) -> torch.Tensor:
    """Read images as RGB and resize them to ``size`` x ``size``.

    Returns:
        Unnormalised ``uint8`` tensor, shape (n, 3, size, size).
    """
    images = [
        np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR))
        for path in paths
    ]
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


BRIGHTNESS_PERCENTILES = (1, 99)
"""Grey-level percentiles of each image mapped to black and white by :func:`rescale_brightness`."""


def rescale_brightness(
    images: torch.Tensor, percentiles: tuple[int, int] = BRIGHTNESS_PERCENTILES
) -> torch.Tensor:
    """Stretch each image's own grey levels to [0, 1], removing brightness differences between images.

    The image's low percentile maps to 0 and its high percentile to 1, with values outside clipped.

    Args:
        images: ``uint8`` tensor, shape (3, H, W) or (n, 3, H, W).
        percentiles: Low and high percentile.

    Returns:
        ``float32`` tensor of the same shape, in [0, 1].
    """
    x = images.float()
    ordered = x.flatten(start_dim=-3).sort(dim=-1).values
    positions = [round(p / 100 * (ordered.shape[-1] - 1)) for p in percentiles]
    low, high = (ordered[..., i, None, None, None] for i in positions)
    return ((x - low) / (high - low).clamp(min=1)).clamp(0, 1)


def normalise(images: torch.Tensor, per_image: bool = True) -> torch.Tensor:
    """Scale ``uint8`` images to [0, 1] and standardise with the ImageNet channel statistics.

    Args:
        images: ``uint8`` tensor, shape (3, H, W) or (n, 3, H, W).
        per_image: Stretch each image's brightness with :func:`rescale_brightness`; if ``False``,
            divide by 255 as for ImageNet.
    """
    scaled = rescale_brightness(images) if per_image else images.float() / 255
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def denormalise(images: torch.Tensor) -> torch.Tensor:
    """Undo :func:`normalise`, returning floats in [0, 1] for display."""
    return (images.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


class MRIDataset(Dataset):
    """Pre-loaded images with optional light augmentation for training.

    Args:
        images: ``uint8`` tensor from :func:`load_images`.
        targets: Class index per image.
        augment: Apply a random horizontal flip and a small random rotation, shift and zoom.
        per_image: Passed to :func:`normalise`.
    """

    def __init__(
        self, images: torch.Tensor, targets: np.ndarray, augment: bool = False, per_image: bool = True
    ) -> None:
        self.per_image = per_image
        self.images = images
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.long)
        self.augment = (
            v2.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalised image (3, H, W) and its class index."""
        image = self.images[i]
        if self.augment is not None:
            image = self.augment(image)
        return normalise(image, self.per_image), self.targets[i]

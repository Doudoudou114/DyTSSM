from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as Data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DataBundle:
    train_size: int
    test_size: int
    train_loader: Data.DataLoader
    test_loader: Data.DataLoader
    full_loader: Data.DataLoader
    full_indices: np.ndarray
    height: int
    width: int
    class_weights: torch.Tensor


def _load_mat(path, variable):
    return sio.loadmat(str(path))[variable]


def _normalize_labels(labels):
    labels = np.asarray(labels)
    labels = np.squeeze(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label map, received {labels.shape}.")
    return labels.astype(np.int64)


def load_data(dataset_name, data_root):
    root = Path(data_root)
    if dataset_name == "hermiston":
        data1 = _load_mat(root / "Hermiston" / "hermiston2004.mat", "HypeRvieW")
        data2 = _load_mat(root / "Hermiston" / "hermiston2007.mat", "HypeRvieW")
        labels = _load_mat(root / "Hermiston" / "label.mat", "label")
        labels = (_normalize_labels(labels) > 0).astype(np.uint8)
    elif dataset_name == "farmland":
        data1 = _load_mat(root / "farm" / "farm06.mat", "imgh")
        data2 = _load_mat(root / "farm" / "farm07.mat", "imghl")
        labels = _load_mat(root / "farm" / "label.mat", "label")
        labels = _normalize_labels(labels).astype(np.uint8)
    elif dataset_name == "river":
        data1 = _load_mat(root / "river" / "river_before.mat", "river_before")
        data2 = _load_mat(root / "river" / "river_after.mat", "river_after")
        labels = _load_mat(root / "river" / "groundtruth.mat", "lakelabel_v1")
        labels = _normalize_labels(labels)
        labels[labels == 255] = 1
        labels = labels.astype(np.uint8)
    elif dataset_name == "bayArea":
        data1 = _load_mat(root / "bayArea" / "Bay_Area_2013.mat", "HypeRvieW")
        data2 = _load_mat(root / "bayArea" / "Bay_Area_2015.mat", "HypeRvieW")
        labels = _load_mat(root / "bayArea" / "bayArea_gtChanges2.mat", "HypeRvieW")
        labels = _normalize_labels(labels).astype(np.int32)
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels,
        ).astype(np.uint8)
    elif dataset_name == "santaBarbara":
        data1 = _load_mat(root / "santaBarbara" / "barbara_2013.mat", "HypeRvieW")
        data2 = _load_mat(root / "santaBarbara" / "barbara_2014.mat", "HypeRvieW")
        labels = _load_mat(root / "santaBarbara" / "barbara_gtChanges.mat", "HypeRvieW")
        labels = _normalize_labels(labels).astype(np.int32)
        labels = np.select(
            [labels == 0, labels == 1, labels == 2],
            [2, 1, 0],
            default=labels,
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data1 = np.asarray(data1, dtype=np.float32)
    data2 = np.asarray(data2, dtype=np.float32)
    if data1.shape != data2.shape:
        raise ValueError(f"Image shapes do not match: {data1.shape} and {data2.shape}.")
    return data1, data2, labels


def normalize_image(image):
    shape = image.shape
    flattened = image.reshape(-1, shape[-1])
    flattened = MinMaxScaler().fit_transform(flattened)
    return flattened.reshape(shape).astype(np.float32)


def apply_pca(image, channels):
    shape = image.shape
    flattened = image.reshape(-1, shape[-1])
    reduced = PCA(n_components=channels, whiten=True).fit_transform(flattened)
    return reduced.reshape(shape[0], shape[1], channels).astype(np.float32)


def _pad_array(array, margin):
    if array.ndim == 3:
        padded = np.zeros(
            (array.shape[0] + margin * 2, array.shape[1] + margin * 2, array.shape[2]),
            dtype=array.dtype,
        )
        padded[margin : margin + array.shape[0], margin : margin + array.shape[1]] = array
        return padded
    if array.ndim == 2:
        padded = np.zeros(
            (array.shape[0] + margin * 2, array.shape[1] + margin * 2),
            dtype=array.dtype,
        )
        padded[margin : margin + array.shape[0], margin : margin + array.shape[1]] = array
        return padded
    raise ValueError(f"Unsupported array shape: {array.shape}.")


def create_patches(image, labels, window_size):
    margin = window_size // 2
    padded_image = _pad_array(image, margin)
    padded_labels = _pad_array(labels, margin)
    patch_count = image.shape[0] * image.shape[1]
    patches = np.zeros(
        (patch_count, window_size, window_size, image.shape[2]),
        dtype=np.float32,
    )
    patch_labels = np.zeros(patch_count, dtype=np.int64)
    patch_index = 0
    for row in range(margin, padded_image.shape[0] - margin):
        for col in range(margin, padded_image.shape[1] - margin):
            patches[patch_index] = padded_image[
                row - margin : row - margin + window_size,
                col - margin : col - margin + window_size,
            ]
            label_patch = padded_labels[
                row - margin : row - margin + window_size,
                col - margin : col - margin + window_size,
            ]
            patch_labels[patch_index] = Counter(label_patch.flatten()).most_common(1)[0][0]
            patch_index += 1
    return patches, patch_labels


def _split_indices(labels, train_ratio):
    indices = np.arange(labels.shape[0])
    try:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=42,
        )
    except ValueError:
        train_indices, test_indices = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=42,
        )
    return train_indices, test_indices


def _add_gaussian_noise(tensor, std):
    return tensor + torch.randn_like(tensor) * std


def generate_data(
    image1,
    image2,
    labels,
    batch_size,
    label_ratio,
    device,
    window_size,
    noise_std,
):
    patches1, patch_labels1 = create_patches(image1, labels, window_size)
    patches2, patch_labels2 = create_patches(image2, labels, window_size)
    if not np.array_equal(patch_labels1, patch_labels2):
        raise RuntimeError("The two temporal patches produced different labels.")

    patches1 = np.transpose(patches1, (0, 3, 1, 2))
    patches2 = np.transpose(patches2, (0, 3, 1, 2))
    valid_mask = patch_labels1 != 2
    patches1 = patches1[valid_mask]
    patches2 = patches2[valid_mask]
    patch_labels = patch_labels1[valid_mask]
    full_indices = np.flatnonzero(labels.flatten() != 2)
    train_indices, test_indices = _split_indices(patch_labels, label_ratio)

    x1_train = torch.from_numpy(patches1[train_indices]).float()
    x2_train = torch.from_numpy(patches2[train_indices]).float()
    y_train = torch.from_numpy(patch_labels[train_indices]).long()
    x1_test = torch.from_numpy(patches1[test_indices]).float()
    x2_test = torch.from_numpy(patches2[test_indices]).float()
    y_test = torch.from_numpy(patch_labels[test_indices]).long()

    x1_train = _add_gaussian_noise(x1_train, noise_std)
    x2_train = _add_gaussian_noise(x2_train, noise_std)
    x1_all = torch.from_numpy(patches1).float()
    x2_all = torch.from_numpy(patches2).float()

    train_dataset = Data.TensorDataset(x1_train, x2_train, y_train)
    test_dataset = Data.TensorDataset(x1_test, x2_test, y_test)
    full_dataset = Data.TensorDataset(x1_all, x2_all)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = Data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    class_counts = Counter(patch_labels[train_indices].tolist())
    weights = []
    for class_index in range(2):
        count = class_counts.get(class_index, 0)
        weights.append(0.0 if count == 0 else len(train_indices) / (2.0 * count))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return DataBundle(
        train_size=len(train_indices),
        test_size=len(test_indices),
        train_loader=train_loader,
        test_loader=test_loader,
        full_loader=full_loader,
        full_indices=full_indices,
        height=labels.shape[0],
        width=labels.shape[1],
        class_weights=class_weights,
    )


def visualize_predictions(original_labels, predictions, indices, output_path):
    full_prediction = np.full(original_labels.size, -1, dtype=np.int16)
    full_prediction[indices] = predictions
    prediction_map = full_prediction.reshape(original_labels.shape)
    visualization = np.zeros((*original_labels.shape, 3), dtype=np.uint8)
    visualization[(original_labels == 0) & (prediction_map == 0)] = [0, 0, 0]
    visualization[(original_labels == 1) & (prediction_map == 1)] = [255, 255, 255]
    visualization[(original_labels == 0) & (prediction_map == 1)] = [255, 0, 0]
    visualization[(original_labels == 1) & (prediction_map == 0)] = [0, 255, 0]
    visualization[original_labels == 2] = [128, 128, 128]
    from matplotlib import pyplot as plt

    plt.imsave(output_path, visualization)
    plt.close()

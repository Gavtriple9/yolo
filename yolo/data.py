import os
import torch
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from yolo import ROOT_DIR


def detection_collate_fn(batch):
    images, targets = [], []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


def create_voc_dataloader(
    batch_size: int, train: bool = True, num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the VOC dataset.
    Args:
        batch_size (int): The batch size for the DataLoader.
        train (bool): If True, create a DataLoader for the training set.
                      If False, create a DataLoader for the validation set.
        num_workers (int): Number of worker threads to use for data loading.

    Returns:
        DataLoader: A DataLoader for the VOC dataset.
    """
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    image_set = "train" if train else "val"
    transform = transforms.Compose(
        [transforms.Resize((416, 416)), transforms.ToTensor()]
    )

    dataset = VOCDetection(
        dataset_dir,
        year="2012",
        image_set=image_set,
        download=True,
        transform=transform,
    )

    if train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn,
    )

    return dataloader

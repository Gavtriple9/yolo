import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

from torchvision.datasets import VOCDetection
from torchvision.transforms import v2

from yolo import ROOT_DIR


def collate_fn(
    data: List[Tuple[torch.Tensor, dict]],
) -> Tuple[torch.Tensor, Tuple[dict]]:
    tensors, targets = zip(*data)
    features = torch.stack(tensors)

    return features, targets


def create_voc_dataloader(
    batch_size: int, train: bool = True, num_workers: int = 8
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
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((448, 448)),
            # v2.ConvertImageDtype(torch.float),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # dataset = VOCDetection(
    #     dataset_dir, image_set="train", year="2012", transform=transform
    # )
    # dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)

    # return dataloader

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
        collate_fn=collate_fn,
    )

    return dataloader

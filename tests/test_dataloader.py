import unittest
import os
import numpy as np
import torch

import matplotlib.pyplot as plt

from yolo.data import create_voc_dataloader
from yolo.utils import draw_bbox_on_image, annos_to_rects, save_image_buf


class TestBoundingBox(unittest.TestCase):
    def setUp(self):
        print("\nCreating dataloader... ")
        self.loader = create_voc_dataloader(64, train=False, num_workers=4)
        print("\033[FCreating dataloader... done")

    def test_plot_some_images(self):
        print("Testing dataloader")
        images: torch.Tensor
        targets: list[dict]
        for batch_index, (images, targets) in enumerate(self.loader):
            annos = targets[0]["annotation"]
            filename = annos["filename"]
            rects = annos_to_rects(annos)
            plt.imshow(images[0].permute(1, 2, 0).numpy())
            rect_image = images[0].permute(2, 1, 0).numpy()

            print(f"rect_image.shape: {rect_image.shape}")

            # for rect in rects:
            #     rect_image = draw_bbox_on_image(rect_image, rect[1])

            save_image_buf(
                rect_image,
                os.path.join("tests", "images", filename),
            )

            if batch_index == 0:
                break


if __name__ == "__main__":
    unittest.main()

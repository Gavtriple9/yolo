import unittest
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import json

from yolo import ROOT_DIR
from yolo.base import Rectangle
from yolo.data import create_voc_dataloader
from yolo.utils import draw_bbox_on_image, objects_to_rects, save_image_buf


class TestObjectParsing(unittest.TestCase):
    def setUp(self):
        single_bbox_path = os.path.join("tests", "data", "single-bbox.json")
        with open(single_bbox_path) as file:
            self.single_bbox_data = json.load(file)["annotation"]

    def test_parse_objects(self):
        objects = self.single_bbox_data["object"]
        rects = objects_to_rects(objects)

        assert len(rects) == 1
        assert rects[0][0] == "tvmonitor"
        assert rects[0][1] == Rectangle((163.5, 152.0), 259.0, 282.0, 1.0)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        print("\nCreating dataloader... ")
        self.loader = create_voc_dataloader(64, train=True, num_workers=4)
        print("\033[FCreating dataloader... done")

    def test_plot_some_images(self):
        print("Testing dataloader")
        features: torch.Tensor
        targets: list[dict]
        for batch_index, (features, targets) in enumerate(self.loader):

            target = targets[0]
            print(f"target: {target}")

            anno = target["annotation"]
            filename = anno["filename"]

            rect_image = np.squeeze(features[0].numpy())
            rect_image = np.transpose(rect_image, [1, 2, 0])

            rects = objects_to_rects(anno["object"])
            print(f"rects: {rects}")

            for rect in rects:
                rect_image = draw_bbox_on_image(rect_image, [rect[1]])

            plt.imshow(rect_image)
            plt.savefig("example.png")

            if batch_index == 0:
                break


if __name__ == "__main__":
    unittest.main()

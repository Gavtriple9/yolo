import unittest
import os
import numpy as np
import PIL.Image as Image

import yolo.utils
from yolo.base.rect import Rectangle
from yolo.base.gridcell import GridCell


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), "../config/default.toml")
        self.width = 500
        self.height = 375
        self.bbox1 = Rectangle((143, 178), 48, 124, 0.1)
        self.bbox2 = Rectangle((338, 236), 297, 241, 0.1)
        self.bbox3 = Rectangle((62, 178), 48, 124, 0.1)
        self.bbox4 = Rectangle((100, 47), 167, 65, 0.1)

    def test_open_toml(self):
        data = yolo.utils.open_toml(self.path)
        self.assertIsInstance(data, dict)

    def test_draw_bboxes_on_image(self):
        input_image = os.path.join(os.path.dirname(__file__), "../images/example.jpg")
        with open(input_image, "rb") as f:
            image_buf = np.array(Image.open(f))

            image_buf = yolo.utils.draw_bbox_on_image(
                image_buf,
                [self.bbox1, self.bbox2, self.bbox3, self.bbox4],
            )
            image = Image.fromarray(image_buf, "RGB")
            image.save("images/test.png")

    def test_generate_grid(self):
        image_size = (450, 513)
        grid_size = (7, 8)

        # Generate grid cells
        grids = yolo.utils.generate_grid_cells(image_size, grid_size)

        # Check if the grid cells are of the correct size
        assert grids.shape == grid_size

        grid: GridCell
        count = 0
        # Check that we didn't lose any pixels
        for grid in grids.flatten():
            count += grid.width * grid.height
        assert count == image_size[0] * image_size[1]

    def test_draw_grid_on_image(self):
        input_image = os.path.join(os.path.dirname(__file__), "../images/example.jpg")
        with open(input_image, "rb") as f:
            image_buf: np.ndarray = np.array(Image.open(f))

        grids = yolo.utils.generate_grid_cells(
            (image_buf.shape[1], image_buf.shape[0]), (7, 7)
        )
        image_buf = yolo.utils.draw_grid_on_image(
            image_buf,
            grids,
        )
        image = Image.fromarray(image_buf, "RGB")
        image.save("images/test2.png")


if __name__ == "__main__":
    unittest.main()

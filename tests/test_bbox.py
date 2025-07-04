import unittest

from yolo.base import Rectangle, Point


class TestBoundingBox(unittest.TestCase):
    def setUp(self):
        """Sets up the following bounding boxes:
                   |
                 __|__
                |  |  |
                |  |  |
        --------|-----|---------->  X
                |  |__|_______
                |__|__|       |
                   |__________|
                   |
                   ↓

                   Y
        """

        self.bbox1 = Rectangle((0, 0), 1, 2, 0.5)
        self.bbox2 = Rectangle((1, 1), 2, 1, 0.5)

    def test_get_top_left(self):
        self.assertEqual(self.bbox1.get_top_left(), Point(-0.5, -1.0))
        self.assertEqual(self.bbox2.get_top_left(), Point(0.0, 0.5))

    def test_get_bottom_right(self):
        self.assertEqual(self.bbox1.get_bottom_right(), Point(0.5, 1.0))
        self.assertEqual(self.bbox2.get_bottom_right(), Point(2.0, 1.5))

    def test_intersects(self):
        self.assertTrue(self.bbox1.intersects(self.bbox2))
        self.assertTrue(self.bbox2.intersects(self.bbox1))

    def test_get_area(self):
        self.assertTrue(self.bbox1.get_area() == 2)
        self.assertTrue(self.bbox2.get_area() == 2)

    def test_get_union_area(self):
        self.assertEqual((self.bbox1.union_area(self.bbox2)), 2 + 2 - 1 / 4)
        self.assertEqual((self.bbox2.union_area(self.bbox1)), 2 + 2 - 1 / 4)

    def test_get_intersection_area(self):
        self.assertEqual(self.bbox1.int_area(self.bbox2), 1 / 4)
        self.assertEqual(self.bbox2.int_area(self.bbox1), 1 / 4)

    def test_get_iou(self):
        self.assertEqual(self.bbox1.iou(self.bbox2), 1 / 15)
        self.assertEqual(self.bbox2.iou(self.bbox1), 1 / 15)


if __name__ == "__main__":
    unittest.main()

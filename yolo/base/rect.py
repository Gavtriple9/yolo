import math
from yolo.base.point import Point


class Rectangle:
    """Bounding box around an object in an image"""

    def __init__(self, p: tuple[float, float], w: float, h: float, c: float):
        """Constructor for a bounding box

        :param p: center of bounding box
        :type p: tuple (float, float)

        :param w: width of bounding box
        :type w: float

        :param h: height of bounding box
        :type h: float

        :param c: confidence of bounding box (will be represented by boldness)
        :type c: float

        :returns: instance of the BoundingBox class
        :rtype: :class:`BoundingBox`
        """
        self.center: Point = Point(p[0], p[1])
        self.w = w
        self.h = h
        self.c = c

    @staticmethod
    def from_corners(top_left: Point, bottom_right: Point) -> "Rectangle":
        return Rectangle(
            Point(
                (top_left.x + bottom_right.x) / 2,
                (top_left.y + bottom_right.y) / 2,
            ),
            abs(bottom_right.x - top_left.x),
            abs(bottom_right.y - top_left.y),
            1.0,
        )

    def __str__(self) -> str:
        return f"({self.x()}, {self.y()}, {self.w}, {self.h}, {self.c})"

    def __repr__(self) -> str:
        return f"({self.x()}, {self.y()}, {self.w}, {self.h}, {self.c})"

    def __eq__(self, other: "Rectangle") -> bool:
        return (
            self.center == other.center
            and self.w == other.w
            and self.h == other.h
            and self.c == other.c
        )

    def __mul__(self, scalar: float) -> "Rectangle":
        s = math.sqrt(math.fabs(scalar))
        w = self.w * s
        h = self.h * s
        return Rectangle((self.center[0], self.center[1]), w, h, self.c)

    def x(self) -> float:
        return self.center.x

    def y(self) -> float:
        return self.center.y

    def get_top_left_pixel(self) -> Point:
        return Point(
            math.floor(self.x() - self.w / 2) + 1,
            math.floor(self.y() - self.h / 2) + 1,
        )

    def get_bottom_right_pixel(self) -> Point:
        return Point(
            math.ceil(self.x() + self.w / 2),
            math.ceil(self.y() + self.h / 2),
        )

    def get_top_left(self) -> Point:
        """
        :returns: top left corner of the bounding box
        :rtype: tuple (float, float)
        """
        return Point(
            self.x() - self.w / 2,
            self.y() - self.h / 2,
        )

    def get_bottom_right(self) -> Point:
        """
        :returns: bottom right corner of the bounding box
        :rtype: tuple (float, float)
        """
        return Point(
            self.x() + self.w / 2,
            self.y() + self.h / 2,
        )

    def get_center(self) -> Point:
        """
        :returns: center of the bounding box
        :rtype: :class:`Point`
        """
        return self.center

    def get_width(self) -> float:
        """
        :returns: width of the bounding box
        :rtype: float
        """
        return self.w

    def get_height(self):
        """
        :returns: height of the bounding box
        :rtype: float
        """
        return self.h

    def get_confidence(self):
        """
        :returns: confidence of the bounding box
        :rtype: float
        """
        return self.c

    def get_area(self):
        """
        :returns: area of the bounding box
        :rtype: float
        """
        return self.w * self.h

    def get_xmin(self):
        """
        :returns: x coordinate of top left corner of the bounding box
        :rtype: float
        """
        return self.get_top_left().x

    def get_ymin(self):
        """
        :returns: y coordinate of top left corner of the bounding box
        :rtype: float
        """
        return self.get_top_left().y

    def get_xmax(self):
        """
        :returns: x coordinate of bottom right corner of the bounding box
        :rtype: float
        """
        return self.get_bottom_right().x

    def get_ymax(self):
        """
        :returns: y coordinate of bottom right corner of the bounding box
        :rtype: float
        """
        return self.get_bottom_right().y

    def intersects(self, other: "Rectangle"):
        """Tests if this bounding box intersects another bounding box

        :returns: True if this bounding box intersects another bounding box
        :rtype: bool
        """
        return (
            self.get_xmin() <= other.get_xmax()
            and self.get_xmax() >= other.get_xmin()
            and self.get_ymin() <= other.get_ymax()
            and self.get_ymax() >= other.get_ymin()
        )

    def contains(self, x, y):
        """Tests if this bounding box contains a given (x,y) point

        :returns: True if this bounding box contains a point
        :rtype: bool
        """
        return (
            self.get_top_left().x < x
            and self.get_top_left().y < y
            and self.get_bottom_right().x > x
            and self.get_bottom_right().y > y
        )

    def on_edge(self, p: Point, delta: float):
        """Tests if a given (x,y) point is on the edge of this bounding box

        :returns: True if a given (x,y) point is on the edge of this bounding box
        :rtype: bool
        """
        inclsv_bb = Rectangle(
            self.x(), self.y(), self.w + delta, self.h + delta, self.c
        )
        exclsv_bb = Rectangle(
            self.x(), self.y(), self.w - delta, self.h - delta, self.c
        )
        return inclsv_bb.contains(p.x, p.y) and not exclsv_bb.contains(p.x, p.y)

    def union_area(self, other: "Rectangle"):
        """Performs the union of this bounding box and another bounding box
        and returns the area of the union

        :returns: the union of this bounding box and another bounding box
        :rtype: float
        """
        if self.intersects(other):
            return self.get_area() + other.get_area() - self.int_area(other)
        else:
            return self.get_area() + other.get_area()

    def int_area(self, other: "Rectangle"):
        """Performs the intersection of this bounding box and another bounding box
        and returns the area of the intersection

        :returns: the intersection of this bounding box and another bounding box
        :rtype: float
        """
        if self.intersects(other):
            top_left = Point(
                max(self.get_top_left().x, other.get_top_left().x),
                max(self.get_top_left().y, other.get_top_left().y),
            )
            bottom_right = Point(
                min(self.get_bottom_right().x, other.get_bottom_right().x),
                min(self.get_bottom_right().y, other.get_bottom_right().y),
            )
            return (bottom_right.x - top_left.x) * (bottom_right.y - top_left.y)
        else:
            return 0.0

    def iou(self, other: "Rectangle"):
        """Calculates the intersection over union of this bounding box and another bounding box

        :returns: intersection over union of this bounding box and another bounding box
        :rtype: float
        """
        return self.int_area(other) / self.union_area(other)

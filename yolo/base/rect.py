import math


class Point:
    """Point in an image"""

    def __init__(self, x: float, y: float):
        """Constructor for a point

        :param x: x coordinate of the point
        :type x: float

        :param y: y coordinate of the point
        :type y: float

        :returns: instance of the Point class
        :rtype: :class:`Point`
        """
        self.x: float = x
        self.y: float = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Point"):
        """Adds two points together

        :param other: point to add
        :type other: :class:`Point`

        :returns: sum of the two points
        :rtype: :class:`Point`
        """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point"):
        """Subtracts two points

        :param other: point to subtract
        :type other: :class:`Point`

        :returns: difference of the two points
        :rtype: :class:`Point`
        """
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float):
        """Multiplies a point by a scalar

        :param scalar: scalar to multiply by
        :type scalar: float

        :returns: product of the point and scalar
        :rtype: :class:`Point`
        """
        return Point(self.x * scalar, self.y * scalar)

    def __div__(self, scalar: float):
        """Divides a point by a scalar

        :param scalar: scalar to divide by
        :type scalar: float

        :returns: quotient of the point and scalar
        :rtype: :class:`Point`
        """
        return Point(self.x / scalar, self.y / scalar)

    def __eq__(self, other: "Point"):
        """Tests if two points are equal

        :param other: point to compare
        :type other: :class:`Point`

        :returns: True if the two points are equal
        :rtype: bool
        """
        return self.x == other.x and self.y == other.y

    def __ne__(self, other: "Point"):
        """Tests if two points are not equal

        :param other: point to compare
        :type other: :class:`Point`

        :returns: True if the two points are not equal
        :rtype: bool
        """
        return not self.__eq__(other)

    def integral(self):
        """Returns the integral of the point

        :returns: integral of the point
        :rtype: :class:`Point`
        """
        return Point(int(self.x), int(self.y))

    def integralize(self):
        """
        Converts the point to an integral point in place
        """
        self.x = int(self.x)
        self.y = int(self.y)

    def real(self):
        """Returns the real value of the point

        :returns: real value of the point
        :rtype: :class:`Point`
        """
        return Point(float(self.x), float(self.y))

    def realize(self):
        """
        Converts the point to a real point in place
        """
        self.x = float(self.x)
        self.y = float(self.y)

    def distance(self, other: "Point"):
        """Calculates the distance between two points

        :param other: point to calculate distance to
        :type other: :class:`Point`

        :returns: distance between the two points
        :rtype: float
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def as_tuple(self):
        """Returns the point as a tuple

        :returns: point as a tuple
        :rtype: tuple (float, float)
        """
        return (self.x, self.y)


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
        self.center: Point = Point(p)
        self.w = w
        self.h = h
        self.c = c

    def __str__(self):
        return f"({self.center.x}, {self.center.y}, {self.w}, {self.h}, {self.c})"

    def __repr__(self):
        return f"({self.center.x}, {self.center.y}, {self.w}, {self.h}, {self.c})"

    def __eq__(self, other: "Rectangle"):
        return (
            self.center == other.center
            and self.w == other.w
            and self.h == other.h
            and self.c == other.c
        )

    def x(self) -> int:
        return int(self.center.x)

    def get_top_left(self) -> Point:
        """
        :returns: top left corner of the bounding box
        :rtype: tuple (float, float)
        """
        return (
            math.floor(self.center.x - self.w / 2) + 1,
            math.floor(self.center.y - self.h / 2) + 1,
        )

    def get_bottom_right(self) -> Point:
        """
        :returns: bottom right corner of the bounding box
        :rtype: tuple (float, float)
        """
        return (
            math.ceil(self.center.x + self.w / 2),
            math.ceil(self.center.y + self.h / 2),
        )

    def get_center(self) -> Point:
        """
        :returns: center of the bounding box
        :rtype: :class:`Point`
        """
        return self.center

    def get_width(self):
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
            self.center.x, self.center.y, self.w + delta, self.h + delta, self.c
        )
        exclsv_bb = Rectangle(
            self.center.x, self.center.y, self.w - delta, self.h - delta, self.c
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
            top_left = (
                max(self.get_top_left().x, other.get_top_left().x),
                max(self.get_top_left().y, other.get_top_left().y),
            )
            bottom_right = (
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

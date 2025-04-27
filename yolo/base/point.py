import math
from cv2.typing import Point as _Point


class Point(_Point):
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

    def __getitem__(self, item: int):
        """Returns the x or y coordinate of the point

        :param item: 0 for x, 1 for y
        :type item: int

        :returns: x or y coordinate of the point
        :rtype: float
        """
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise IndexError("Point index out of range")

    def __len__(self):
        return 2

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

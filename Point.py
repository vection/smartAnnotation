from math import sqrt


class Point:
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @x.setter
    def x(self, x_value):
        self._x = x_value

    @y.setter
    def y(self, y_value):
        self._y = y_value

    def __repr__(self):
        return "({}, {})".format(self._x, self._y)

    def __str__(self):
        return "({}, {})".format(self._x, self._y)

    def __sub__(self, other):
        """
        finds the Euclidean distance between two points
        :param other:
        :return: distance
        """
        d_square = (self._x - other.x) ** 2 + (self._y - other.y) ** 2
        return sqrt(d_square)
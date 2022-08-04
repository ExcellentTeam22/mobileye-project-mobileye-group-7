from typing import NamedTuple

class Rectangle(NamedTuple):
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max (self.right - self.left, 0)

    @property
    def height(self) -> float:
        return max (self.top - self.bottom, 0)

    @property
    def area(self) -> float:
        return self.width * self.height
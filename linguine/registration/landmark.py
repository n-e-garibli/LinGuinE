# Copyright AstraZeneca 2026
"""This file contains a dataclass which defines a landmark coordinate."""

from dataclasses import dataclass


@dataclass
class LandmarkCoordinate:
    """A dataclass for storing the coordinate of a landmark."""

    x: int | None = None
    y: int | None = None
    z: int | None = None

    def __post_init__(self):
        self.is_valid: bool = True
        if any([(v is None or v < 0) for v in [self.x, self.y, self.z]]):
            # this instance of LandmarkCoordinate does not contain
            # a valid coordinate due to the presence of a null/negative value.
            self.is_valid = False

# Copyright AstraZeneca 2026
"""An abstract class for extracting a set of 3D landmark coordinates from an input.
The input can be a path to a file or an array like structure with the image itself.
This is the parent class for extracting landmarks as common reference points that
registration algorithms rely on."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from linguine.registration.landmark import LandmarkCoordinate


class PointExtractor(ABC):
    """An abstract class for extracting a set of 3D landmark coordinates from an input"""

    @property
    @abstractmethod
    def valid_ids(self) -> set[str]:
        """This property defines the set of valid landmark names that the point extractor
        can load/compute."""
        pass

    @abstractmethod
    def extract_points(
        self,
        input: dict[str, Any],
        include_ids: Iterable[str] | None = None,
        *args,
        **kwargs,
    ) -> dict[str, LandmarkCoordinate]:
        """Extract landmark points from the input data.

        Args:
            input: The input data dictionary containing the data to extract points from.
            include_ids: An optional iterable of landmark IDs to extract. If None, all
                valid landmarks will be extracted. Any ID not in self.valid_ids will be
                ignored.

        Returns:
            A dictionary mapping landmark names to their coordinates. If a landmark
            is not found in the input, its value will be an empty LandmarkCoordinate.

            Example:
                {
                    "landmark1": LandmarkCoordinate(x=0, y=0, z=0),
                    "landmark2": LandmarkCoordinate(x=1, y=0, z=2),
                    "landmark3": LandmarkCoordinate()  # Not found
                }
        """
        pass

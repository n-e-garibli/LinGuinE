# Copyright AstraZeneca 2026
"""Contains a CSVPointExtractor which interfaces with a CSV file to extract points."""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from linguine.constants import FILE_ID
from linguine.registration.landmark import LandmarkCoordinate
from linguine.registration.point_extractors import PointExtractor


class CSVPointExtractor(PointExtractor):
    """Simple point extractor that reads landmarks from a CSV."""

    @property
    def valid_ids(self) -> set[str]:
        all_ids = self.landmark_df[self.landmark_id_colname].unique()
        return set(all_ids)

    def __init__(
        self,
        landmark_csv_path: str,
        img_identifier_colname: str = FILE_ID,
        landmark_id_colname: str = "name",
        x_colname: str = "x",
        y_colname: str = "y",
        z_colname: str = "z",
    ):
        """Constructs the class.

        Args:
            landmark_csv_path: path to a csv file that contains landmark information.
            img_identifier_colname: name of the column in the csv that identifies the image.
                This identifier will be the expected input into the extract_points() method of this
                extractor.
            landmark_id_colname: name of the column in the csv which identifies the landmark.
            x_colname: name of the column in the csv which identifies the x coordinate.
            y_colname: name of the column in the csv which identifies the y coordinate.
            z_colname: name of the column in the csv which identifies the z coordinate.
        """
        self.landmark_df = pd.read_csv(landmark_csv_path)
        column_names = list(self.landmark_df.columns)
        for colname in [
            img_identifier_colname,
            landmark_id_colname,
            x_colname,
            y_colname,
            z_colname,
        ]:
            assert colname in column_names, f"Colname {colname} not found in csv."

        self.landmark_id_colname = landmark_id_colname
        self.img_identifier = img_identifier_colname
        self.x_colname = x_colname
        self.y_colname = y_colname
        self.z_colname = z_colname

    def extract_points(
        self,
        input: dict[str, Any],
        include_ids: Iterable[str] | None = None,
    ) -> dict[str, LandmarkCoordinate]:
        """Extract landmark points from the CSV file the class was instantiated with for the given input.

        Args:
            input: The input data dictionary. Must contain a FILE_ID key that matches
                the identifier column in the CSV file.
            include_ids: An optional iterable of landmark IDs to extract. If None, all
                valid landmarks will be extracted.

        Returns:
            A dictionary mapping landmark names to their coordinates. Missing landmarks
            will have empty LandmarkCoordinate objects.
        """
        relevant_df = self.landmark_df[self.landmark_df[self.img_identifier] == input[FILE_ID]]
        if include_ids is not None:
            relevant_df = relevant_df[relevant_df[self.landmark_id_colname].isin(include_ids)]

        output = {}
        for _, row in relevant_df.iterrows():
            output[row[self.landmark_id_colname]] = LandmarkCoordinate(
                x=row[self.x_colname],
                y=row[self.y_colname],
                z=row[self.z_colname],
            )

        for landmark_name in self.valid_ids:
            if include_ids is not None:
                if landmark_name not in include_ids:
                    continue
            elif landmark_name not in output:
                output[landmark_name] = LandmarkCoordinate()
        return output

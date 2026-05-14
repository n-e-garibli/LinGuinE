# Copyright AstraZeneca 2026
import math

import numpy as np
import scipy.ndimage as ndi
import torch

from linguine.config import MaskSamplingConfig
from linguine.utils.bounding_boxes import get_bounding_box


def find_mask_center(
    array: np.ndarray | torch.Tensor,
    trim_channel: bool = True,
) -> tuple[int, ...]:
    """For a given mask, return the coordinate of a positive pixel that is
    closest to mask center using the chamfer method..

    Args:
        array: Input n-dim mask array.
        trim_channel: Return only the last three or two channels for 3D and 2D images.

    Returns:
        tuple[int, ...]: Calculated center of mask.
    """
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    # Get locations of ones in this array across each dim.
    positives: tuple[np.ndarray, ...] = np.where(array > 0)

    # No positive pixels, return an empty coordinate.
    if len(positives[0]) == 0:
        return ()

    transformed: np.ndarray = ndi.distance_transform_cdt(array)
    unique = np.unique(transformed)
    unique = unique[~np.isin(unique, [0])]  # removes 0 value
    locs: tuple[np.ndarray, ...] = np.where(transformed == np.random.choice(unique[-1:]))
    # If multiple pixels are of the highest summed distance, we use the median.
    median: int = len(locs[0]) // 2
    closest = tuple([loc[median] for loc in locs])

    # Check if the sampled pixel is inside input mask array.
    if array[closest] == 0:
        raise ValueError("Center is found outside of mask.")
    if trim_channel:
        if len(closest) > 3:
            return closest[-3:]
    return closest


def uniform_sampling(
    n_samples: int,
    array_3D: torch.Tensor,
) -> list[tuple[int, int, int]]:
    """Function used to uniformly sample points without replacement
    from a 3D array with option to not sampling points at the edges of array.

    Args:
        n_samples: Number of points to sample.
        array_3D: 3D array to sample from.
    """
    x_indices, y_indices, z_indices = torch.where(array_3D == 1)
    random_indices: torch.Tensor = torch.randperm(len(x_indices))[:n_samples]
    sampled = []

    # Get the coordiante data.
    for random_coordinate_i in random_indices:
        x, y, z = (
            x_indices[random_coordinate_i].item(),
            y_indices[random_coordinate_i].item(),
            z_indices[random_coordinate_i].item(),
        )
        sampled.append((x, y, z))
    return sampled


def sample_clicks(
    binary_array: torch.Tensor,
    config: MaskSamplingConfig,
    spacing: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    """
    Sample clicks from the provided binary array, as per the method and configurations in config.

    Args:
        binary_array: tensor of binary array from which to sample clicks
        config: MaskSamplingConfig object of configuration to use
        spacing: spacing of each dimension.

    Returns:
        List of sampled clicks
    """

    if config.method == "uniform":
        # NOTE: Will need to be converted to a tensor in the future
        return uniform_sampling(n_samples=config.num_samples, array_3D=binary_array)

    spacing = torch.tensor(spacing, device=binary_array.device)

    if config.method in ["quadratic", "normal"]:
        clicks = sample_from_distributions(binary_array, config=config, spacing=spacing)

    if config.method in ["fixed_number_clicks", "fixed_click_distance"]:
        clicks = sample_fixed_clicks(binary_array, config=config, spacing=spacing)

    # NOTE: Moving away from torch here for registration compatibility. Will be updated once registration also uses torch.
    return [tuple(map(int, point.tolist())) for point in clicks]


def sample_from_distributions(
    binary_array: torch.Tensor,
    config: MaskSamplingConfig,
    spacing: torch.Tensor,
) -> torch.Tensor:
    """
    Sample clicks from a binary_array using a weighted
    distribution (quadratic or normal) based on distances from the component's center.
    Euclidean distances are used to calculate weights

    Args:
        binary_array: tensor from which to sample clicks from
        config: MaskSamplingConfig object of configuration to use
        spacing: spacing of each dimension.

    Returns:
        Tensor of sampled clicks
    """

    coords = torch.nonzero(binary_array, as_tuple=False)
    if config.num_samples >= coords.numel():
        return coords

    mean_coord = torch.mean(coords.float(), dim=0)
    distances = get_euclidean_distances(coords, mean_coord, spacing=spacing)

    center = coords[torch.argmin(distances)]

    distances = get_euclidean_distances(coords, center, spacing)

    if config.method == "quadratic":
        probabilities = quadratic_distribution(distances)

    elif config.method == "normal":
        probabilities = normal_distribution(distances)

    sampled_indices = torch.multinomial(probabilities, num_samples=config.num_samples, replacement=False)
    sampled_coords = coords[sampled_indices]

    if center not in sampled_coords:
        sampled_coords[-1] = center

    return sampled_coords


def update_shift_and_number(
    distance: int, num_clicks: int, click_spacing: int, method: str
) -> tuple[torch.Tensor, int]:
    """
    Calculate the shift to center and adjust the number of clicks if needed based on sampling method

    Args:
        distance: distance between clicks
        num_clicks: number of clicks to sample
        click_spacing: distance, in voxels, between clicks.
        method: sampling method used, as specified in MaskSamplingConfig

    Returns:
        Tuple of tensor representing shift of every click and (updated) number of clicks to sample

    """
    if click_spacing != 0:
        shift = math.floor(distance % click_spacing) / 2

        if (shift == 0) and (method == "fixed_click_distance"):
            num_clicks += 1

    else:
        num_clicks = distance
        shift = 0

    return shift, num_clicks


def sample_fixed_clicks(binary_array: torch.Tensor, config: MaskSamplingConfig, spacing) -> list[tuple[int, int, int]]:
    """
    Generates a grid of candidate clicks from the bounding box of the binary_array. Each grid point is shifted
    to (approximately) the cell-center.
    (optionally, a replacement search is performed).

    Args:
        binary_array (torch.Tensor): 3D binary tensor.
        config: MaskSamplingConfig object of configuration to use
        spacing: spacing of each dimension.

    Returns:
        Tensor of sampled clicks
    """

    coords = torch.nonzero(binary_array, as_tuple=False)

    bb = get_bounding_box(binary_array, as_mask=False)

    x_dist, y_dist, z_dist = (
        bb.x_max + 1 - bb.x_min,
        bb.y_max + 1 - bb.y_min,
        bb.z_max + 1 - bb.z_min,
    )

    if config.method == "fixed_number_clicks":
        num_voxels_per_click = tuple(
            [
                math.floor(dist / config.num_clicks_per_dimension[idx])
                for idx, dist in enumerate([x_dist, y_dist, z_dist])
            ]
        )
        (
            num_x_clicks,
            num_y_clicks,
            num_z_clicks,
        ) = config.num_clicks_per_dimension

    elif config.method == "fixed_click_distance":
        num_voxels_per_click = config.num_voxels_per_click_per_dimension

        num_x_clicks, num_y_clicks, num_z_clicks = (
            math.floor(dist / num_voxels_per_click[idx]) for idx, dist in enumerate([x_dist, y_dist, z_dist])
        )

    x_shift, num_x_clicks = update_shift_and_number(x_dist, num_x_clicks, num_voxels_per_click[0], method=config.method)
    y_shift, num_y_clicks = update_shift_and_number(y_dist, num_y_clicks, num_voxels_per_click[1], method=config.method)
    z_shift, num_z_clicks = update_shift_and_number(z_dist, num_z_clicks, num_voxels_per_click[2], method=config.method)

    clicks = []

    for xs in range(0, max(1, num_x_clicks)):
        for ys in range(0, max(1, num_y_clicks)):
            for zs in range(0, max(1, num_z_clicks)):
                point = (
                    int(bb.x_min + xs * max(1, num_voxels_per_click[0]) + x_shift),
                    int(bb.y_min + ys * max(1, num_voxels_per_click[1]) + y_shift),
                    int(bb.z_min + zs * max(1, num_voxels_per_click[2]) + z_shift),
                )

                if binary_array[point] == 1:
                    clicks.append(torch.tensor(point, device=coords.device))

                elif config.replacement is True:
                    replacement_point = nearby_search(
                        point=point,
                        coords=coords,
                        search_size=num_voxels_per_click,
                        spacing=spacing,
                    )

                    if replacement_point is not None:
                        clicks.append(replacement_point)

    return clicks


def quadratic_distribution(distances: torch.Tensor) -> torch.Tensor:
    """
    Compute quadratic weights for the input distances such that the minimum is half the largest distance

    Args:
        distances: Tensor of values from which to calculate probabilities

    Returns:
        Tensor of probabilities corresponding to each input distance
    """

    max_dist = torch.max(distances)
    probabilities = (max_dist / 2 - distances) ** 2
    return probabilities / probabilities.sum()


def normal_distribution(distances: torch.Tensor) -> torch.Tensor:
    """
    Compute sampling probabilities based on a normal distribution with extended tails beyond the edge of the lesion.

    Args:
        distances: Tensor of values from which to calculate probabilities

    Returns:
        Tensor of probabilities corresponding to each input distance
    """

    std_dev = torch.mul(distances, 2).float().std()
    sigma = std_dev if distances.numel() > 1 else 1.0
    weights = torch.exp(-0.5 * (distances.float() / sigma) ** 2)
    return weights / weights.sum()


def get_euclidean_distances(coordinates: torch.Tensor, point: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
    """
    Compute the Euclidean distances between each coordinate and a given point,
    applying a scaling factor per dimension.

    Args:
        coordinates: coordinates to use.
        point: coordinate from which to calculate the distances.
        spacing: spacing of each dimension.

    Returns:
        The euclidean distance to the centre for each given coordinate.
    """

    scaled_diff = (coordinates - point) * spacing
    distances = torch.sqrt(torch.sum(scaled_diff**2, dim=1))

    return distances


def nearby_search(
    point: tuple[int, int, int],
    coords: torch.Tensor,
    search_size: tuple[int, int, int],
    spacing: torch.Tensor,
) -> torch.Tensor | None:
    """
    Given a tensor of foreground coordinates, a point, and a search_size for each dimension,
    returns the closest foreground point within the cuboid defined by the search size
    If no coordinate lies within this cuboid, returns None.

    Args:
        point: Coordinate of the point around which to search.
        coords: List of foreground coordinates.
        search_size: Distance to next point in each dimension. Search space is defined by half this value.
        spacing: Spacing between each dimension

    Returns:
        The coordinate that is closest to the point among those in the search space, or None if none exist.
    """

    point = torch.tensor(point, device=coords.device)
    search_size = torch.tensor(search_size, device=coords.device) / 2

    lower_bound = point - search_size
    upper_bound = point + search_size

    mask = ((coords >= lower_bound) & (coords <= upper_bound)).all(dim=1)
    filtered_coords = coords[mask]

    if filtered_coords.shape[0] > 0:
        return find_closest_coordinate(filtered_coords, point, spacing)
    else:
        return None


def find_closest_coordinate(coordinates: torch.Tensor, point: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
    """
    Find the coordinate closest to the given point, using weighted Euclidean distance.

    Args:
        coordinates: Coordinates to analyse
        point: Point to which the closest coordinate is to be found
        spacing: Spacing between each dimension

    Returns:
       The coordinate closest the point
    """

    distances = get_euclidean_distances(coordinates, point, spacing)
    min_index = torch.argmin(distances)
    return coordinates[min_index]

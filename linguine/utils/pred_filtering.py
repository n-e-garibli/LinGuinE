# Copyright AstraZeneca 2026
"""
If this seems like overkill, it probably is. Most functions were pulled from a more elaborate
internal repo where all this jazz is needed and I didn't have time to simplify them for
their purpose here.
"""

from dataclasses import dataclass

import cc3d
import numpy as np
import torch

from linguine.utils.misc import find_closest_point


def filter_prediction(
    clicks: tuple[int, int, int] | list[tuple[int, int, int]],
    pred: np.ndarray,
    dist_threshold: int = 10,
    torch_array: bool = True,
    merge_bbox: bool = True,
    bbox_expansion: int = 0,
    device: str = "cuda",
    return_tensor: bool = False,
) -> tuple[np.ndarray | torch.Tensor, list[list[tuple[int, int]]]]:
    """
    Filter input (pred) array using clicks. Connected-components in the
    prediction will be calculated, and only CCs with clicks inside them (or near
    them) will be preserved.

    Args:
        clicks: A click or a list of clicks.
        pred: Ouput array from a model.
        dist_threshold: If the clicks are outside predictions, the nearest CC
            within this threshold will be kept.
            1. If dist_threshold is negative, this function will be skipped.
                Original prediction will be returned and no bounding box will be
                found.
            2. If set to 0, only click-containing predictions will be kept, and
                the algorithm will not find nearby predictions.

        torch_array: Use pytorch as array operation module instead of numpy.
        merge_bbox: Whether to merge overlapping bounding boxes.
        bbox_expansion: The number of pixels by which to expand the bounding boxes
        in each direction before merging. If negative, this will not be applied.
        device: Device to use for torch array operations.
        return_tensor: If True, the output will be a torch tensor. Otherwise, it
            will be a numpy array.

    Returns:
        np.ndarray: Filtered prediction (could be an all-zero array).
        list[list[tuple[int, int]]]: Bounding boxes for each click input.
    """
    # Also skip calculation if clicks are empty.
    if dist_threshold < 0 or not clicks:
        return (pred, [])
    # Sanity checks and conversions.
    if isinstance(clicks, tuple):
        clicks = [clicks]
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    assert all([len(click) == pred.ndim for click in clicks]), "Click dimension does should match array dimension."
    # list to hold bounding boxes.
    bboxes: list[list[tuple[int, int]]] = []

    # Create the connected-component array.
    cc: np.ndarray = cc3d.connected_components(pred).astype(np.uint8)  # Originally int16, and is not accepted by torch.

    if torch_array:
        # malloc an output array which WILL BE MUTATED (as an optimization).
        mutable_output: torch.Tensor = torch.zeros(pred.shape, dtype=torch.uint8).to(device)
        cc: torch.Tensor = torch.from_numpy(cc).to(device)
    else:
        mutable_output: np.ndarray = np.zeros(pred.shape, dtype=np.uint8)

    for click in clicks:
        # NOTE: this mutates output array.
        bbox: list[tuple[int, int]] = _find_cc(
            click=click,
            cc=cc,
            output=mutable_output,
            dist_threshold=dist_threshold,
            known_bboxes=bboxes,
            torch_array=torch_array,
        )

        if bbox is not None:
            if bbox_expansion > 0:
                bbox = [
                    (
                        max(dim_bounds[0] - bbox_expansion, 0),
                        min(dim_bounds[1] + bbox_expansion, max_dim),
                    )
                    for dim_bounds, max_dim in zip(bbox, pred.shape)
                ]
            bboxes.append(tuple(bbox))

    if torch_array and not return_tensor:
        mutable_output = mutable_output.cpu().numpy()
    if merge_bbox:
        bboxes = merge_3d_bboxes(bboxes)
    return mutable_output, bboxes


def _find_cc(
    click: tuple[int, int, int],
    cc: np.ndarray | torch.Tensor,
    output: np.ndarray | torch.Tensor,
    dist_threshold: int = 10,
    known_bboxes: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None,
    torch_array: bool = True,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    For a single location/click, modify output array such that the CC
    including, or is near the click will be marked as 1.

    Args:
        click: Click coordinate.
        cc: Connected-component array built by "cc3d.connected_components()".
        output: Mutable output array.
        dist_threshold: If the clicks are outside predictions, the nearest CC
            within this threshold will be kept.
        known_bboxes: Existing bounding boxes from previous clicks. Used as a
            reference to prevent repetitive calculation.
        torch_array: Use pytorch as array operation module instead of numpy.

    Returns:
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]: Bounding box's
            min and max values on three axes.

    NOTE: Argument "output" is mutated in this function in-place.
    """
    ccid: int = cc[click]
    cal_module = torch if torch_array else np
    bbox: list[tuple[int, int]] = None

    # The click is inside a connected-component.
    if ccid > 0:
        # Find locations need modifying.
        positives: tuple[np.ndarray, ...] | tuple[torch.Tensor, ...] = cal_module.where(cc == ccid)
        # Get bounding box of the modified region.
        bbox = tuple(
            (
                cal_module.min(x).tolist(),
                cal_module.max(x).tolist(),
            )
            for x in positives
        )
        # This bbox was seen before so we can skip the rest of this function.
        if known_bboxes and bbox in known_bboxes:
            return None
        # Mutate the output array. We don't make a copy to save runtime.
        output[positives] = 1

    # The click is outside a connected-component.
    # We then find the closest one (depends on arg "dist_threshold").
    else:
        if dist_threshold > 0:
            # Slower alternative: positives_outside = np.where(cc > 0)
            positives_outside: tuple[np.ndarray, ...] | tuple[torch.Tensor, ...] = _find_positives_within_range(
                click=click,
                dist=dist_threshold,
                cc=cc,
                torch_array=torch_array,
            )
            if len(positives_outside[0]) == 0:  # No positive point within range.
                return bbox
            closest: tuple[int, int, int] = find_closest_point(
                positives=positives_outside,
                target=click,
            )
            dist: float = np.linalg.norm(np.array(closest) - np.array(click))
            if dist <= dist_threshold:
                positives = cal_module.where(cc == cc[closest])
                bbox = tuple(
                    (
                        cal_module.min(x).tolist(),
                        cal_module.max(x).tolist(),
                    )
                    for x in positives
                )
                if known_bboxes and bbox in known_bboxes:
                    return None
                output[positives] = 1

    return bbox


def _find_positives_within_range(
    click: tuple[int, int, int],
    dist: int,
    cc: np.ndarray | torch.Tensor,
    torch_array: bool = True,
) -> tuple[np.ndarray, ...] | tuple[torch.Tensor, ...]:
    """
    Within a range, find and return positive voxel coordinates.
    An alternative would be "np.where(cc > 0)" without specifiying a range,
    but that approach would be slower.

    Args:
        click: Center of the distance-finding.
        dist: Max span with click as center.
        cc: Array to find positives from. Returned coordinates are based on this array.
        torch_array: Use pytorch as array operation module instead of numpy.

    Returns:
        tuple[np.ndarray, ...]: 3D coordinates of positive voxels.
    """
    assert cc.ndim == len(click) == 3, "Dimension other than 3 is not supported."
    cal_module = torch if torch_array else np

    # Cropping indices.
    lower: int = dist // 2
    higher: int = dist - lower
    x_min, y_min, z_min = (max(0, click_coor - lower) for click_coor in click)
    x_max, y_max, z_max = (min(max_dim, click_coor + higher) for max_dim, click_coor in zip(cc.shape, click))

    # Now we find positives on a much smaller cube.
    cropped: np.ndarray | torch.Tensor = cc[x_min:x_max, y_min:y_max, z_min:z_max]
    cropped_positives: tuple[np.ndarray, ...] | tuple[torch.Tensor, ...] = cal_module.where(cropped > 0)

    # Add back offsets of these coordinates.
    for delta, dim_coor in zip([x_min, y_min, z_min], cropped_positives):
        dim_coor += delta

    return cropped_positives


Range3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass
class BBox:
    box: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    box_id: int = None
    x_group: int = None
    y_group: int = None
    z_group: int = None


def merge_1D(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Merge a list of overlapping intervals, and return the result.
    E.g. [(1,4), (2, 5), (7, 10)] -> [(1, 5), (7, 10)].
    """
    assert intervals, "Got an empty list."
    sorted_invervals: list[tuple[int, int]] = sorted(intervals, key=lambda x: x[0])

    results: list[tuple[int, int]] = []

    start, end = sorted_invervals[0]
    for n_start, n_end in sorted_invervals[1:]:
        if n_start > end:
            results.append((start, end))
            start, end = n_start, n_end
        else:  # I.e., n_start <= end
            end = max(end, n_end)

    results.append((start, end))

    return results


def merge_1D_get_index(intervals: list[tuple[int, int]]) -> list[int]:
    """
    Merge a list of overlapping intervals, but return their group indices only.
    E.g. [(1,4), (2, 5), (7, 10)] -> [0, 0, 1].
    """
    assert intervals, "Got an empty list."
    sorted_invervals: list[tuple[int, int]] = sorted(intervals, key=lambda x: x[0])

    current_index: int = 0
    results: list[int] = [current_index]

    start, end = sorted_invervals[0]
    for n_start, n_end in sorted_invervals[1:]:
        if n_start > end:
            current_index += 1
            end = n_end
        else:  # I.e., n_start <= end
            end = max(end, n_end)
        results.append(current_index)
    return results


def merge_1D_get_index_counter(intervals: list[tuple[int, int]]) -> list[int]:
    """
    Merge a list of overlapping intervals, but return group index counters.
    E.g. [(1,4), (2, 5), (7, 10)] -> [(1, 5), (7, 10)]; and the group indices are
    [0, 0, 1]. This function will return counters for each group: [2, 1].
    """
    assert intervals, "Got an empty list."
    sorted_invervals: list[tuple[int, int]] = sorted(intervals, key=lambda x: x[0])

    current_counter: int = 1
    results: list[int] = []

    start, end = sorted_invervals[0]
    for n_start, n_end in sorted_invervals[1:]:
        if n_start > end:
            results.append(current_counter)
            current_counter = 1
            end = n_end
        else:  # I.e., n_start <= end
            end = max(end, n_end)
            current_counter += 1

    results.append(current_counter)
    return results


def grouping_recursion(merged: list[Range3D], grouped_bboxes: dict[int, BBox]):
    """
    Recursion method to find 3D bounding boxes.

    The idea is that we group a series of 3D ranges by their X, Y and Z values. Each group may
    contain just one Range3D object: this means this Range3D is isolated from other Range3Ds;
    alternatively, a group may contain multiple of them, and they can **potentially** be merged.
    However, this is not entirely true because we also need to recompute due to the possibility
    that those isolated ranges may serve as "linkers" connecting otherwise discrete groups (reason
    being they are generated at the same depth of recursion). We therefore need to go one depth
    deeper for those seemingly grouped ranges until they cannot be divided further. We can then
    merge them at that point.

    Args:
        merged: Reference to a list that restores results.
        grouped_bboxes: dictionary of available bounding bounding boxes. Values are
            instances of BBox class.
    """
    # Keep a record of passed box ids. Used in the stop-condition of recursion.
    id_record: set[int] = set(grouped_bboxes.keys())

    # We now group these Range3D (incorporated within BBox instances) by their X, Y and Z ranges.
    # Sort by x coors.
    x_sorted_bboxes: list[BBox] = sorted(grouped_bboxes.values(), key=lambda bbox: bbox.box[0][0])
    x_sorted_coors: list[Range3D]
    x_sorted_i: list[int]
    (
        x_sorted_coors,
        x_sorted_i,
    ) = zip(  # Both ranges and ids are sorted the same way.
        *[(b.box[0], b.box_id) for b in x_sorted_bboxes]
    )
    # Update bbox X grouping. The function `merge_1D_get_index` returns grouping indices.
    for sorted_i, grouping_index in zip(x_sorted_i, merge_1D_get_index(x_sorted_coors)):
        grouped_bboxes[sorted_i].x_group = grouping_index  # Modify instance's grouping info.

    # Sort by Y coors.
    y_sorted_bboxes: list[BBox] = sorted(grouped_bboxes.values(), key=lambda bbox: bbox.box[1][0])
    y_sorted_coors, y_sorted_i = zip(*[(b.box[1], b.box_id) for b in y_sorted_bboxes])
    for sorted_i, grouping_index in zip(y_sorted_i, merge_1D_get_index(y_sorted_coors)):
        grouped_bboxes[sorted_i].y_group = grouping_index
    # Sort by Z coors.
    z_sorted_bboxes: list[BBox] = sorted(grouped_bboxes.values(), key=lambda bbox: bbox.box[2][0])
    z_sorted_coors, z_sorted_i = zip(*[(b.box[2], b.box_id) for b in z_sorted_bboxes])
    for sorted_i, grouping_index in zip(z_sorted_i, merge_1D_get_index(z_sorted_coors)):
        grouped_bboxes[sorted_i].z_group = grouping_index

    # Assign bbox ids to BBox instances. The keys are XYZ grouping ids, and values
    # are bboxes with those id sets. Only ranges within the same group may get merged.
    mergable: dict[tuple[int, int, int], list[int]] = {}
    for bbox_instance in grouped_bboxes.values():
        assert (
            bbox_instance.x_group is not None
            and bbox_instance.y_group is not None
            and bbox_instance.z_group is not None
        ), "Not all bbox was visited."
        bbox_group: tuple[int, int, int] = (
            bbox_instance.x_group,
            bbox_instance.y_group,
            bbox_instance.z_group,
        )
        if bbox_group not in mergable:
            mergable[bbox_group] = []
        mergable[bbox_group].append(bbox_instance.box_id)

    # Now iterate over all groups.
    for group in mergable.values():
        # Isolated bbox. We add them to result directly.
        if len(group) == 1:
            merged.append(grouped_bboxes[group[0]].box)

        # Otherwise, this group contains multiple Range3Ds.
        else:
            # This is the only group we got and id_record shows that after previous
            # grouping operations it remained the same. This indicates this group cannot
            # be divided further, so we can merge them add add the result to the output.
            if len(mergable) == 1 and set(group) == id_record:
                # Extract raw ranges.
                box_coors: list[Range3D] = [grouped_bboxes[bi].box for bi in group]
                # Merge them on each dimension.
                merged.append(
                    tuple(
                        merge_1D([x[0] for x in box_coors])
                        + merge_1D([x[1] for x in box_coors])
                        + merge_1D([x[2] for x in box_coors])
                    )
                )
            # There are multiple groups (or after isolation check the result got changed).
            # This means we need to further check if this group can be divided, and therefore
            # recurse.
            else:
                grouping_recursion(merged, {i: grouped_bboxes[i] for i in group})


def merge_3d_bboxes(bboxes: list[Range3D]) -> list[Range3D]:
    """
    Merge a list of 3D bounding boxes by calling a recursion function.

    E.g.
        input_bboxes = [
            # Overlapping x, y, z.
            ((0, 1), (0, 1), (0, 1)),
            ((0, 2), (0, 2), (0, 2)),
            ((2, 5), (2, 5), (2, 5)),

            # Overlapping x, y.
            ((10, 11), (10, 11), (10, 11)),
            ((10, 11), (10, 11), (20, 30)),

            # Overlapping x.
            ((20, 40), (50, 60), (60, 70)),
            ((25, 30), (40, 50), (40, 50)),

            # Overlapping x, z.
            ((50, 70), (90, 100), (80, 90)),
            ((55, 60), (70, 80), (90, 100)),

            # Overlapping y.
            ((80, 90), (110, 120), (110, 120)),
            ((100, 110), (105, 130), (130, 140)),

            # Corner case. Overlapping x, y, and the first two entries have
            # overlapping z; and z of the third entry "connects" the first two.
            ((118, 139), (146, 167), (167, 188)),
            ((119, 140), (175, 196), (172, 193)),
            ((120, 141), (154, 175), (138, 159)),

            # Overlapping x, y, z.
            ((500, 501), (511, 514), (500, 510)),
            ((501, 502), (513, 515), (490, 520)),
            ((502, 503), (514, 516), (480, 530)),

            # Overlapping x, y, z.
            ((1000, 1010), (1011, 1019), (1000, 1010)),
            ((1009, 1020), (1019, 1040), (1009, 1020)),
            ((1019, 1030), (1040, 1060), (1019, 1030)),
        ]

        expected_results = [
            # Only x-y-z-overlapping ones can be merged.
            ((0, 5), (0, 5), (0, 5)),
            # The rest are not merged.
            ((10, 11), (10, 11), (10, 11)),
            ((10, 11), (10, 11), (20, 30)),
            ((20, 40), (50, 60), (60, 70)),
            ((25, 30), (40, 50), (40, 50)),
            ((50, 70), (90, 100), (80, 90)),
            ((55, 60), (70, 80), (90, 100)),
            ((80, 90), (110, 120), (110, 120)),
            ((100, 110), (105, 130), (130, 140)),
            ((118, 139), (146, 167), (167, 188)),
            ((119, 140), (175, 196), (172, 193)),
            ((120, 141), (154, 175), (138, 159)),
            ((500, 503), (511, 516), (480, 530)),
            ((1000, 1030), (1011, 1060), (1000, 1030)),
        ]

    Args:
        bboxes: A list of 3D bounding boxes.

    Returns:
        list[Range3D]: Merged
            bounding boxes.
    """
    # Results get stored here.
    merged: list[Range3D] = []

    # Sort by x values.
    # Note that a de-duplication process is necessary, otherwise it will fail
    # isolation check.
    sorted_bboxes: list[Range3D] = sorted(list(set(bboxes)), key=lambda x: x[0][0])
    # Create a series of BBox instances.
    bbox_list: list[BBox] = [BBox(box=bbox, box_id=i) for i, bbox in enumerate(sorted_bboxes)]
    # Add hash table of those instances.
    grouped_bboxes = {bbox.box_id: bbox for bbox in bbox_list}
    # Call the recursion function.
    grouping_recursion(merged, grouped_bboxes)
    # Return sorted results.
    return sorted(merged)

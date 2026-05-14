# LinGuinE
**L**ongitud**IN**al **GUI**daNce **E**stimation

LinGuinE is a plug-and-play framework for segmenting volumetric medical images longitudinally developed at AstraZeneca.
For more information about the method, please see our preprint: https://arxiv.org/abs/2506.06092.

Note that there are additional features present in the codebase that were not discussed in the paper. This includes some heuristic classes to help identify disappearing lesions, improve propagation accuracy, other "boosting" techniques, and support for the propagation of bounding boxes.

## Updates and Release Timeline
1. Provisional MICCAI acceptance - 07/05/2026 ✅
2. Initial code release - 14/05/2026 ✅
3. UniToChest associated longitudinal mask release - estimated 22/05/2026
4. LinGuinE generated masks for public datasets release - estimated 01/06/2026
5. Support for clicks as source scan guidance - TBD
6. Wrappers for some publicly available segmentation models - TBD
7. Innate support for ROI based unguided segmentation models - TBD

## Installation

### Standard install

```
pip install git+https://github.com/n-e-garibli/LinGuinE.git
```

### Development install

Clone the repository and install in editable mode with development dependencies:

```
git clone https://github.com/n-e-garibli/LinGuinE.git
cd LinGuinE
```
```
python -m venv .venv
```
```
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```
```
pip install -e ".[dev]"
```

## Quick Start Guide

For new users, we recommend starting with a simple configuration and gradually customizing components as needed. The basic workflow involves:

1. **Prepare your data dictionaries** with the required keys (see input format below)
2. **Create a wrapper for your segmentation model** following the `AbstractInferer` interface in linguine/inferers
3. **Configure LinGuinE** using the LinguineConfig class
4. **Run the longitudinal segmentation** using LinguineDatasetProcessor

## Obtaining Segmentations with LinGuinE

### The Input Data Format
The input into LinGuinE is a list of data dictionaries, where each dictionary represents a sample to potentially be segmented. The dictionary can contain the following keys:

**Required keys:**
- **"patient_id"**: a string identifier for the patient that the sample belongs to. Must be present for all samples.
- **"file_id"**: a string identifier for this sample. Must be unique across all input data dicts and present for all samples.
- **"image"**: a path to the image to be loaded. Must be present for all samples.
- **"timepoint"**: an identifier for the timepoint that this sample was taken at (eg: "week_12").

**Required for evaluation mode:**
- **"label"**: a path to the segmentation label corresponding to the image. Must be present for one sample per patient, and this will be the sample that guidance is propagated from.

**Optional keys:**
- **"use_as_source"**: a boolean flag to specify that this particular scan should be used as the source scan for longitudinal propagation. Only one scan per patient can have this flag set to `True`. If no scan has this flag, LinGuinE will automatically select the earliest timepoint scan with a valid label as the source scan.
- **"clicks"**: a list of clicks in the form `[(x, y, z),]` to use as source scan guidance if a segmentation label is not available for the source scan. At the moment this will only work properly if you have one tumour per study.
- **"ts"**: a path to a pre-computed TotalSegmentator (https://github.com/wasserth/totalsegmentator) segmentation file. Can be used to aide registration depending on the method used. 

**Example data dictionary:**
```python
{
    "patient_id": "patient_001",
    "file_id": "patient_001_week_0",
    "image": "/path/to/scan.nii.gz",
    "label": "/path/to/segmentation.nii.gz",  # Optional if predict_only=True
    "timepoint": "week_0",
}
```

**Evaluation vs Prediction Mode:**
Please note that by default LinGuinE runs in evaluation mode - this means that it expects a label to be present for every sample to compute various performance metrics. You can turn this behaviour off using the `predict_only` parameter in the configuration, which is useful for segmenting scans without ground truth.

### The Segmentation Model
LinGuinE is model agnostic, meaning that any segmentation model can be used to generate segmentations. We do not provide a baseline segmentation model as part of this package. To use your model with LinGuinE, you must create a wrapper for it following the `AbstractInferer` interface. An instance of that can then be passed into LinGuinE.

**For ROI-based Models:**
If your segmentation model operates on fixed-size regions of interest (ROIs) rather than full images, you can use the `ROIInferer` abstract base class. This class automatically handles:
- Extracting ROIs around guidance clicks
- Converting guidance data to ROI coordinate space
- Mapping ROI predictions back to full image dimensions

To use ROI-based inference, inherit from `ROIInferer` instead of `AbstractInferer` and implement the `roi` property (defining the expected ROI size) and the `infer_on_roi` method.

**Filtering Predictions to the Relevant Connected Component:**
We recommend making use of the `filter_prediction` utility (found in `linguine/utils/pred_filtering.py`) inside your inferer's inference method. `filter_prediction` runs connected-component analysis on the model output and retains only the component(s) that contain or are nearest to the provided click(s), discarding unrelated detections.

### Minimal Example

```python
import numpy as np
import torch
from linguine.config import AnalysisConfig, LinguineConfig, RegistrationConfig
from linguine.dataset_processor import LinguineDatasetProcessor
from linguine.inferers.base_inferer import AbstractInferer
from linguine.utils.bounding_boxes import BBox3D
from linguine.utils.pred_filtering import filter_prediction


# Step 1 — wrap your segmentation model in an AbstractInferer
class MyInferer(AbstractInferer):
    def __init__(self, model):
        self.model = model

    def infer(
        self,
        img: torch.Tensor,
        fg_clicks: list[tuple[int, int, int]],
        bg_clicks: list[tuple[int, int, int]],
        bboxes: list[BBox3D],
        return_probs: bool = False,
        filter_pred: bool = True,
    ) -> np.ndarray:
        # Run your model and obtain a binary prediction array.
        # Adapt to your model's API
        raw_pred = self.model(img, fg_clicks, bg_clicks, bboxes)  

        # Keep only the connected component(s) nearest to the guidance clicks.
        if filter_pred and fg_clicks:
            raw_pred, _ = filter_prediction(clicks=fg_clicks, pred=raw_pred)

        return raw_pred


# Step 2 — build the list of data dictionaries (one entry per scan)
data_dicts = [
    {
        "patient_id": "patient_001",
        "file_id": "patient_001_week_0",
        "image": "/data/patient_001/week_0.nii.gz",
        "label": "/data/patient_001/week_0_seg.nii.gz",  # source scan label
        "timepoint": "week_0",
    },
    {
        "patient_id": "patient_001",
        "file_id": "patient_001_week_6",
        "image": "/data/patient_001/week_6.nii.gz",
        "timepoint": "week_6",
    },
    {
        "patient_id": "patient_001",
        "file_id": "patient_001_week_12",
        "image": "/data/patient_001/week_12.nii.gz",
        "timepoint": "week_12",
    },
]

# Step 3 — configure LinGuinE
cfg = LinguineConfig(
    save_predictions=True,
    predict_only=True,  # set to False if labels are available for all scans and you want metrics.
    device="cpu",
    registration=RegistrationConfig(registrator="aruns"),
    analysis=AnalysisConfig(iteration_mode="FROM_ONE_TIMEPOINT"),
)

# Step 4 — run
inferer = MyInferer(model=...)  # pass your instantiated model here
processor = LinguineDatasetProcessor(
    cfg=cfg,
    inferer=inferer,
    save_folder="/output/results",
    data_dicts=data_dicts,
)
processor.process_dataset()
# Predictions are saved to /output/results/pred_<file_id>.nii.gz
# A summary CSV is saved to /output/results/results.csv
```

## Customising LinGuinE

There are multiple components that can be customised to determine how the longitudinal segmentation is obtained:

1. **The registrator** - LinGuinE supports both **point-set registration** (using anatomical landmarks) and **image registration** (working directly with image intensities). Any registrator can be used as long as it follows the appropriate interface (PointSetRegistrator or ImageRegistrator).

2. **The point extractor** - This determines the landmarks used for registration when using point-set registration methods. You can create your own custom PointExtractor to get whatever landmarks you want, however you want.

3. **The prompt selector** - We provide prompt selectors that can choose better clicks from candidate locations in the tumour (the center may not be optimal due to the tumour changing and registration imperfections). Available selectors include threshold-based and unguided model-based approaches. When no prompt selector is configured, only the center click from the source lesion will be propagated.

4. **The disappearance detector** - This component determines whether a predicted lesion should be kept or filtered out, helping detect cases where lesions have disappeared between timepoints. We provide size-based and intensity-based disappearance detectors that can be combined.

5. **The image loader** - Custom image loaders can be provided to handle different image formats or preprocessing requirements. The image loader must simply turn information in the "image" and "label" keys of the input data dictionary into two Monai MetaTensors representing the image and label. Designing custom image loaders gives more flexibility on the format of your input data dict - there is no reason why the images/labels must be represented by paths to the file.

### Registration Methods: Point-Set vs Image Registration

LinGuinE supports two fundamentally different approaches to image registration:

#### Image Registration (Intensity-Based)  
**How it works:** Directly compares images to find optimal alignment. No landmarks required; think Demons, VoxelMorph, GradICON, etc. 

#### Point-Set Registration (Landmark-Based)
**How it works:** Uses anatomical landmarks as reference points to compute transformations between images. Landmarks are extracted from both source and target images, then a mathematical transformation (rigid or non-rigid) is computed to align the corresponding landmarks. This is very fast once landmarks are available and is robust to intensity differences between scans - We support Arun's rigid registration method and Thin Plate Spline Registration (we find that this can deform things too much, so make sure to regularise)

For point-set registration, LinGuinE uses anatomical landmarks for image registration between timepoints. You will need to generate your landmarks first - see tutorials/registration for an example of how you may do this. You can then use the CSV by specifying `point_extractor_csv` in the registration configuration

### The Output
LinGuinE predictions will be saved as compressed nifti files in the `save_folder` folder specified in the input. Each prediction will be named "pred_{file_id}.nii.gz" based on the identifier present in the data dictionary for that sample. There will also be a results.csv file containing the details of every propagation, such as the coordinates of propagated prompts and metrics (if not in evaluation mode).

## Repository Structure

A quick guide to find key files.

```
LinGuinE/
├── README.md                    # Project documentation
├── pyproject.toml               # Python package configuration
├── linguine/                    # Main package source code
│   ├── config.py                # Configuration classes
│   ├── dataset_processor.py     # Main processing pipeline for a set of input data dicts.
│   ├── propagator.py            # Core LinGuinE click propagation logic
│   ├── disappearance_detectors/ # Disappearance detection components
│   │   └── detectors.py         # Size and intensity-based filtering
│   ├── image_loaders/           # Image loading utilities
│   │   └── base_image_loader.py # Basic monai based loader
│   ├── inferers/
│   │   ├── base_inferer.py      # Abstract inferer interface
│   │   ├── roi_inferer.py       # Abstract inferer interface for ROI-based models
│   │   └── boosted_inferers.py  # Enhanced inference methods
│   ├── prompt_selectors/        # Prompt selection components
│   │   ├── base_ps.py           # Abstract base class
│   │   ├── threshold_ps.py      # Intensity-based prompt selection
│   │   └── unguided_ps.py       # Model-based prompt selection
│   ├── registration/            # Image registration components
│   │   ├── landmark.py          # Landmark coordinate dataclass
│   │   ├── point_extractors/    # Anatomical landmark extractors
│   │   └── registrators/        # Point set registration methods
│   ├── study_segmentors/        # Segmentors for a single longitudinal study.
│   │   ├── base_segmentor.py    # Abstract base class
│   │   ├── from_one_tp.py       # From one timepoint propagation
│   │   └── chain.py             # Chain (autoregressive) propagation
│   └── utils/                   # General utilities
├── tests/                       # Unit tests
└── tutorials/                   # Notebooks with demos on how to use LinGuinE
```

## LinGuinE Configuration Options

LinGuinE is highly configurable through the `LinguineConfig` class. Here's a comprehensive overview of all available options with detailed explanations:

### Main Configuration (`LinguineConfig`)

These are the top-level settings that control the overall behavior of LinGuinE:

- **`save_predictions: bool`** - When set to `True`, LinGuinE will save all segmentation predictions as compressed NIfTI files (.nii.gz) in your output directory. Each file will be named "pred_{file_id}.nii.gz". This is useful for visual inspection and further analysis of results.

- **`save_with_original_affine: bool`** - When set to `True`, saved predictions will be reoriented to match the affine matrix of the original input image. Requires `save_predictions=True`.

- **`predict_only: bool`** - Set to `True` when you don't have ground truth segmentation labels available. This disables all evaluation metrics (like Dice scores) and allows LinGuinE to work purely in prediction mode. Made for scenarios where you're segmenting new, unlabeled scans.

- **`device: str`** - Specifies which computational device to use (e.g., "cuda:0" for GPU, "cpu" for CPU).

- **`patient_ids: list[str] | None`** - If provided, LinGuinE will only process the specified patients. Useful for debugging or processing specific subsets of your data. Set to `None` to process all patients.

### Registration Configuration (`RegistrationConfig`)

These settings control how LinGuinE aligns images from different timepoints:

- **`point_extractor_csv: str | None`** - Path to a pre-computed CSV file containing anatomical landmarks. **Only required for point-set registration methods** For image registration methods, this parameter is ignored.

- **`registrator: str`** - The registration method to use for aligning images:
  - **"aruns"**: Arun's rigid registration method. Fast and works well when anatomy doesn't change much between scans.
  - **"tps"**: Thin Plate Spline registration. More flexible, can handle some anatomical deformation between timepoints.
  - **"perfect"**: Cheat mode that uses ground truth tumor centers (only for research/debugging).

The publicly available GradICON models are also supported, if you pip install their package.

- **`tps_lambda: float`** - Regularization parameter for TPS registration (only used when `registrator="tps"`). Higher values make the transformation smoother but less flexible. We find large values (> 1000) yield better performance.

- **`crop_foreground: bool`** - Whether to crop to foreground region for certain registrators (gradICON likes this). For registrators where masked preprocessing is applied, this will crop to the bounding box around the mask.

### Analysis Configuration (`AnalysisConfig`)

These settings determine how LinGuinE propagates segmentations across multiple timepoints:

- **`iteration_mode: str`** - The strategy for temporal propagation:
  - **"FROM_ONE_TIMEPOINT"**: Uses the first possible scan as a template and propagates to ALL other timepoints directly.
  - **"CHAIN"**: Propagates step-by-step through time (scan1→scan2→scan3→...). More robust to gradual changes but errors can accumulate.
  - **"CHAIN_WITH_RESAMPLING"**: Like CHAIN but regenerates guidance clicks at each step from the previous prediction. This is even more sensitive to segmentation error accumulation, but can better handle changing objects. Used for the autoregressive analysis in our paper.

- **`prompt_to_propagate: str`** - The type of guidance prompt to propagate:
  - **"click"** (default): Propagates single-point clicks as guidance. Compatible with prompt selector filtering.
  - **"bbox"**: Propagates 3D bounding boxes as guidance. The corners of the bounding box are propagated through registration and reconstructed in the target space.
  - **"bbox_2d"**: Propagates 2D bounding boxes as guidance. Similar to `"bbox"` but constructs a per-slice 2D bounding box in the target space.

- **`boosting: str | None`** - Advanced techniques to improve segmentation quality, including:
  - **None**: No boosting, use model predictions as-is.
  - **"basic"**: Resamples a click at the center of the initial model prediction and uses that as the final input. Can help bring the propagated click closer to the lesion center. This is the mode discussed and evaluated in our paper.
  - **"resample_additive"**: Same as above but provides both the original clicks and the resampled click as input.
  - **"resample_merge_probs"**: Obtains predictions with individual clicks and merges the probability maps to get the final segmentation.
  - **"perturb_ensemble"**: Perturbs the original click(s) and produces an ensemble prediction from the perturbed variants.
  - **"click_ensemble"**: Obtains individual predictions for each click and returns an ensemble.
  - **"orientation_ensemble"**: Produces an ensemble of predictions obtained from different image orientations.

- **`min_pred_size_mm: float`** - Filters out predicted lesions smaller than this diameter in the axial plane. Setting this to >0 will use the SizeFilter disappearance detector.

- **`seed: int`** - Random seed for reproducible results.

- **`forward_in_time_only: bool`** - When set to `True`, LinGuinE will only propagate to timepoints that come **after** the source scan. When `False` (default), it propagates to all other timepoints regardless of temporal order.

- **`num_perturbations: int`** - If greater than 0, the guidance click is perturbed this many times and the resulting predictions are compared to the un-perturbed prediction. This is used to estimate prediction confidence. Defaults to 0 (disabled).

- **`timepoint_flags: list[str] | None`** - Optional list of timepoint identifier patterns to use when sorting scans by timepoint. Each pattern should be a string like `"week_"`, `"cycle_"`, etc. The pattern will be used to extract a numeric timepoint value from scan identifiers (e.g., `"week_4"` → 4). If not provided, a default set of common patterns will be used: `["week_", "end_of_cycle_", "disease_assessment_", "cycle_", "fu_", "follow_up_", "exam_", "timepoint_"]`. Use this argument if your dataset has timepoints named in a different format.

### Prompt Selection Configuration (`PromptSelectionConfig`)

The PromptSelector chooses optimal clicks from candidate locations.

- **`type: str | None`** - The method for selecting optimal propagated clicks:
  - **None**: Use only the center click from the source lesion without any selection logic. Simple and fast.
  - **"threshold"**: Use intensity-based criteria to select the best click from candidate locations. Selects clicks in regions within specified intensity thresholds.
  - **"unguided"**: Use the segmentation model to evaluate candidate clicks and select the one most likely to produce a good segmentation based on unguided predictions.

- **`n_clicks: int`** - Maximum number of clicks to use during prompt selection. The selector will choose up to this many clicks from the candidate locations. More clicks can improve accuracy in larger lesions but increase computation time. Typical values: 1-5.

- **`l_threshold: float | int`** - The lower intensity threshold value (only used when `type="threshold"`). Candidate clicks in image regions with intensity below this value are penalized or rejected.

- **`u_threshold: float | int`** - The upper intensity threshold value (only used when `type="threshold"`). Candidate clicks in image regions with intensity above this value are penalized or rejected. 

- **`mask_sampling: MaskSamplingConfig`** - Controls how candidate clicks are generated from the source lesion mask for the prompt selector to evaluate.

## Contributing

Contributions are welcome! Feel free to open a pull request, particularly if you'd like to add support for a new custom component (e.g. a new registrator, prompt selector, disappearance detector, or image loader).

## Citation

If you use LinGuinE in your research, please cite our preprint. We will update this citation once LinGuinE is published.

```bibtex
@misc{garibli2026linguine,
  title         = {LinGuinE: Longitudinal Guidance Estimation for Volumetric Tumour Segmentation},
  author        = {Garibli, Nadine and Csiba, Bence and Sidiropoulos, Kostantinos and Wei, Yi and Patwari, Mayank},
  year          = {2026},
  eprint        = {2506.06092},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2506.06092},
  primaryClass  = {eess.IV},
  url={https://arxiv.org/abs/2506.06092},
}
```

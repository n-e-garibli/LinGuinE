# Benchmarking

This directory contains various assets for benchmarking and to facilitate the reproducibility of the results in our paper (https://arxiv.org/abs/2506.06092.)

## UniToChest Associated Labels

One of our test datasets in our paper was the publicly available **UniToChest** - https://zenodo.org/records/5797912. This dataset contains some longitudinal CT studies (i.e. some patients have more than 1 timepoint imaged). However, the label masks provided are binary. This means that there is no way to distinguish between individual lesions or track them across time. Therefore to evaluate LinGuinE on our dataset we had to make the labels longitudinal, and are now releasing these labels (unitochest_labels.zip) to facilitate further research on longitudinal segmentation and tracking. As of May 2026, This is one of the only public datasets out there that has tumour segmentations with consistent labelling across time.

### How we created these masks

The original binary masks were split by connected component - each component was assumed to be a distinct tumour. The masks were relabelled so that each distinct tumour had its own label value - up to 10 largest connected components were kept per mask (some scans had 100+). An experienced radiologist then visualised these masks and made notes on his perceived tumour associations across each longitudinal study considered. This meant that we had notes of the form "tumour 2 in exam 1 and tumour 7 in exam 3 are the same tumour". We ignored scenarios where tumours coalesced, where he was unsure, or where there were too many lesions to make the task reasonable. Not all tumours present will be annotated - further work can be done to expand this dataset if anyone has the time.

Please note that in the original dataset the masks were saved as jpgs. I hated that, so our version are niftis. 

### How to relate them back to the original images. 

Each mask is saved as ``{patient_id}_{exam_number}.nii.gz``. The original dataset is structured quite intuitively by patient and exam number - you should be able to find the corresponding dcm image quite easily. Note that since I had to make the labels out of jpgs, some of the metadata may not be exactly the same as the corresponding image. As far as I can tell the metadata is correct, and I haven't had issues getting the images/labels to align when loading them in various ways. If you have any trouble with this or finding the corresponding images in the first place, reach out to me and I will be able to share the images in nifti format with the exact same metadata as the labels and same file names.

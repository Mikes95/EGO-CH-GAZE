# EGO-CH-GAZE

![Add an Image Here](./img/back.jpg) <!-- Replace "link_to_image.png" with the actual image URL or file path -->

Welcome to the official GitHub repository for "Learning to Detect Attended Objects in Cultural Sites with Gaze Signals and Weak Object Supervision."

## Overview

To address the challenging problem of attended object detection in cultural sites, we have curated a unique dataset of egocentric images captured by subjects while visiting cultural sites. These images offer a glimpse into the visual experiences of museum-goers and come equipped with labels for artworks and objects that have captured the subjects' attention.

In our work, we present two innovative approaches for attended object detection at various weakly supervised levels. These approaches strike a balance between performance and the amount of supervision required, as demonstrated by our experiments. Specifically, we introduce:

### Box Coordinates Regressor

- Code can be found in the `/code/BBox_regressor` directory.
- This code loads a ResNet model, preprocesses data, and trains a deep learning model for gaze estimation using a Gaussian-based approach. This approach is particularly effective when bounding boxes around attended objects are available for training.

### Fully Convolutional Attended Object Detection

For a detailed explanation of this method, please visit our companion repository at [https://github.com/fpv-iplab/WS-Attended-Object-Detection](https://github.com/fpv-iplab/WS-Attended-Object-Detection).

### Unsupervised Methods

You can explore our unsupervised methods by examining the code in the `/code/Unsupervised` directory. This directory contains various scripts for performing inference using SAM, InSPyReNet, and U-2-Net.



## Cite us

If you find our work useful, please cite the following paper:

```bibtex
@article{10.1145/3647999,
author = {Mazzamuto*, Michele and Ragusa*, Francesco and Furnari*, Antonino and Farinella*, Giovanni Maria},
title = {Learning to Detect Attended Objects in Cultural Sites with Gaze Signals and Weak Object Supervision},
year = {2024},
issue_date = {September 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {17},
number = {3},
issn = {1556-4673},
url = {https://doi.org/10.1145/3647999},
doi = {10.1145/3647999},
articleno = {35},
numpages = {21},
keywords = {Cultural sites, wearable devices, gaze, object detection}
}

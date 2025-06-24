
# Synthetic RGB-D Camera Model
This repository provides tools for enhancing synthetically generated RGB-D images by simulating realistic sensor phenomena such as noise, occlusions, and aligning depth to RGB frames.

## Features
- Noise Simulation: Add lateral and axial noise, drop pixels based on surface angle and color.
- Depth Filtering: Local minimum filtering to refine depth maps.
- Depth-to-Point Cloud: Back-project depth images to 3D point clouds and reproject into RGB frame.
- Alignment: Full pipeline to align depth maps to RGB images, with optional interpolation.
- Batch Processing: Orchestrate end-to-end processing over a directory of images.

## Requirements

- Python 3.7+
- NumPy
- SciPy
- OpenCV
- Open3D

Install dependencies with:

```pip install numpy scipy opencv-python open3d```

## Usage
### Preprocess Depth

```
from preprocessing import PreprocessingManager
proc = PreprocessingManager(params)
depth_noisy = proc.get_processed_image(depth_img, rgb_img)
```

### Align Depth to RGB

```from projection import ProjectionManager
proj = ProjectionManager(rgb_params, depth_params, T)
aligned = proj.get_aligned_depth_img(depth_noisy, rgb_img)``` 

### Batch Processing

```from processor import ImageProcessor
processor = ImageProcessor('params.json', 'input_dir', 'output_dir')
processor.process_and_save_all_images()```

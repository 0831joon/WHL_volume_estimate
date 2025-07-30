# LiDAR Volume Calculation and ROI Extraction
# This repository contains a set of scripts to process LiDAR point cloud data for volume difference calculations before and after a digging operation. The pipeline involves data preprocessing, region of interest (ROI) extraction, point cloud registration, and volume difference computation.
## File Structure
```
// calculate_volume_ws/
// │
// ├── pointclouds/
// │   ├── after_digging_preprocessed.pcd
// │   ├── after_digging_reference_frame.npy
// │   ├── after_digging_roi.npy
// │   ├── before_digging_preprocessed.pcd
// │   ├── before_digging_reference_frame.npy
// │   ├── before_digging_roi.npy
// │
// ├── After_digging.sh
// ├── After_digging_processing.py
// ├── Before_digging.sh
// ├── Before_digging_processing.py
// └── Preprocessing.py
```
## Usage
### 1. Preprocess LiDAR Data
// To begin the process, run the shell script for either the "before" or "after" digging data. The script will preprocess the raw data from the specified ROS bag and output a downsampled point cloud file.
// For "After Digging":
```bash
// bash After_digging.sh
```
// For "Before Digging":
```bash
// bash Before_digging.sh
```
### 2. Region of Interest (ROI) Extraction
// After preprocessing the point clouds, the next step is to extract the region of interest (ROI) for both before and after the digging data. This step is done automatically within the processing scripts. The results are saved as `.npy` files for later use.
### 3. Volume Difference Calculation
// Once the ROIs are extracted and the point clouds are registered, the volume difference between the "before" and "after" states is computed using the extracted ROIs. This is done by applying a grid-based method to compute the volume of material displaced during the digging operation.
### 4. Visualization
// Optional visualization is available for most steps, such as showing the point clouds, regions of interest, and the volume difference (if the `--visualize` flag is set).
---
## Requirements
// - Python 3.x
// - Open3D
// - scikit-learn
// - ROS2 (for bag file reading)
// You can install the required Python packages via `pip`:
```bash
// pip install open3d scikit-learn
```
---
## Script Descriptions
### Preprocessing.py
// This script processes raw LiDAR data from a ROS bag and outputs a downsampled PCD file. It uses DBSCAN clustering to filter out noise and apply voxel downsampling.
### After_digging_processing.py
// This script extracts the region of interest (ROI) from the "after digging" point cloud and calculates the volume difference by comparing the "before" and "after" point clouds.
### Before_digging_processing.py
// This script extracts the ROI from the "before digging" point cloud and prepares the data for further processing.
### Shell Scripts (`.sh`)
// - `After_digging.sh`: Executes the preprocessing and processing for "after digging" data.
// - `Before_digging.sh`: Executes the preprocessing and processing for "before digging" data.
---
### Note:
// Ensure that the paths to the ROS bag files and the output directories are correctly set before running the shell scripts.
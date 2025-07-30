#!/bin/bash

# Preprocessing for LiDAR raw data (rosbag2 -> PCD conversion + downsampling)
AFTER_DIR="../rosbag_wheel/after_digging/single_pose/rosbag2_2025_07_09-14_02_18"
# AFTER_DIR="../rosbag_wheel/after_digging/single_pose/rosbag2_2025_07_09-14_04_44"
AFTER_PCD_DIR="./pointclouds/after_digging_preprocessed.pcd"
AFTER_REF_DIR="./pointclouds/after_digging_reference_frame.npy"
python3 Preprocessing.py --input $AFTER_DIR --output-pcd $AFTER_PCD_DIR --output-ref $AFTER_REF_DIR

# ROI extraction for after digging & volume difference calculation
python3 After_digging_processing.py
#!/bin/bash

# Preprocessing for LiDAR raw data (rosbag2 -> PCD conversion + downsampling)
BEFORE_DIR="../rosbag_wheel/before_digging/single_pose/rosbag2_2025_07_09-13_48_04"
# BEFORE_DIR="../rosbag_wheel/before_digging/single_pose/rosbag2_2025_07_09-13_52_11"
BEFORE_PCD_DIR="./pointclouds/before_digging_preprocessed.pcd"
BEFORE_REF_DIR="./pointclouds/before_digging_reference_frame.npy"
python3 Preprocessing.py --input $BEFORE_DIR --output-pcd $BEFORE_PCD_DIR --output-ref $BEFORE_REF_DIR

# ROI extraction for before digging
python3 Before_digging_processing.py
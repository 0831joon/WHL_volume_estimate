#!/usr/bin/env python3

from turtle import st
import open3d as o3d
import numpy as np
import argparse
import time
from sklearn.cluster import DBSCAN
from collections import defaultdict

def crop_point_cloud_by_radius(pcd, radius):
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < radius
    return pcd.select_by_index(np.where(mask)[0])

def main():
    parser = argparse.ArgumentParser()
    # Add arguments for input and output paths
    parser.add_argument('--input_pcd', type=str, default='./pointclouds/before_digging_preprocessed.pcd', help='Path to before_digging_preprocessed.pcd')
    parser.add_argument('--input_T', type=str, default='./pointclouds/before_digging_reference_frame.npy', help='Path to before_digging_reference_frame.npy')
    parser.add_argument('--output_roi',type=str, default='./pointclouds/before_digging_roi.npy', help='Path to before_digging_roi.npy')
    # ICP parameters
    parser.add_argument('--threshold', type=float, default=1.0, help='ICP correspondence threshold')
    parser.add_argument('--radius', type=float, default=15.0, help='Cropping radius (meters)')


    # ROI extraction parameters
    parser.add_argument("--roi-radius", type=float, default=30.0, help="ROI radius for plane extraction")
    parser.add_argument("--cluster-eps", type=float, default=0.5, help="DBSCAN cluster epsilon")
    parser.add_argument("--cluster-min-points", type=int, default=20, help="DBSCAN cluster min points")

    # Volume calculation parameters
    parser.add_argument("--grid-size", type=float, default=0.1, help="Grid size for volume calculation")

    # Visualization flag
    parser.add_argument("--visualize", action="store_true", help="Visualize point cloud")
    args = parser.parse_args()



    start_time = time.time()    
    before_pcd = o3d.io.read_point_cloud(args.input_pcd)
    before_T = np.load(args.input_T)
    before_pcd.transform(np.linalg.inv(before_T))
    
    # Crop point cloud
    before_pcd = crop_point_cloud_by_radius(before_pcd, args.radius)
    before_points = np.asarray(before_pcd.points)
    before_mask = (before_points[:, 0] > 0) & (before_points[:, 0] < args.roi_radius) & (before_points[:, 1] > 0) & (before_points[:, 1] < args.roi_radius) & (before_points[:, 2] > 0) & (before_points[:, 2] < args.roi_radius)
    before_roi_candidate_points = before_points[before_mask]
    before_roi_pcd = o3d.geometry.PointCloud()
    before_roi_pcd.points = o3d.utility.Vector3dVector(before_roi_candidate_points)
    # DBSCAN clustering
    before_clustering = DBSCAN(eps=args.cluster_eps, min_samples=args.cluster_min_points).fit(before_roi_candidate_points)
    before_labels = before_clustering.labels_
    before_unique_labels = set(before_labels)
    before_min_dist = float('inf')
    before_best_label = -1
    for label in before_unique_labels:
        if label == -1:
            continue
        cluster_points = before_roi_candidate_points[before_labels == label]
        center = cluster_points.mean(axis=0)
        dist = np.linalg.norm(center)
        if dist < before_min_dist:  
            before_min_dist = dist
            before_best_label = label
    
    # best_label이 가장 원점에 가까운 cluster
    before_roi_points = before_roi_candidate_points[before_labels == before_best_label]
    np.save(args.output_roi, before_roi_points)

    if args.visualize:
        before_roi_pcd.points = o3d.utility.Vector3dVector(before_roi_points)
        before_roi_pcd.paint_uniform_color([0.3, 0.3, 1.0])
        o3d.visualization.draw_geometries([before_roi_pcd], window_name="Region of Interest")

    time_end = time.time()
    print(f"Extract Region of Interest \t completed in {time_end - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

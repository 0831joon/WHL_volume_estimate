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

def run_icp(source_pcd, target_pcd, threshold=1.0, init=np.eye(4)):
    # target_pcd must have normals
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30)
    )
    target_pcd.orient_normals_towards_camera_location()
    
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse

def main():
    parser = argparse.ArgumentParser()
    # Add arguments for input and output paths
    parser.add_argument('--input_before_pcd', type=str, default='./pointclouds/before_digging_preprocessed.pcd', help='Path to before_digging_preprocessed.pcd')
    parser.add_argument('--input_after_pcd', type=str, default='./pointclouds/after_digging_preprocessed.pcd', help='Path to after_digging_preprocessed.pcd')
    parser.add_argument('--input_before_T', type=str, default='./pointclouds/before_digging_reference_frame.npy', help='Path to before_digging_reference_frame.npy')
    parser.add_argument('--input_after_T', type=str, default='./pointclouds/after_digging_reference_frame.npy', help='Path to after_digging_reference_frame.npy')
    parser.add_argument('--input_before_roi', type=str, default='./pointclouds/before_digging_roi.npy', help='Path to before_digging_roi.npy')
    parser.add_argument('--output_after_roi', type=str, default='./pointclouds/after_digging_roi.npy', help='Path to after_digging_roi.npy')
    
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

    time_start1 = time.time()    
    before_pcd = o3d.io.read_point_cloud(args.input_before_pcd)
    after_pcd = o3d.io.read_point_cloud(args.input_after_pcd)
    before_T = np.load(args.input_before_T)
    after_T = np.load(args.input_after_T)
    before_pcd.transform(np.linalg.inv(before_T))
    after_pcd.transform(np.linalg.inv(after_T))

    # Crop by radius (centered at before_pcd center)
    before_pcd = crop_point_cloud_by_radius(before_pcd, args.radius)
    after_pcd = crop_point_cloud_by_radius(after_pcd, args.radius)
    before_pcd.paint_uniform_color([0.3, 0.3, 1.0])
    after_pcd.paint_uniform_color([1.0, 0.3, 0.3])
    
    # if args.visualize:
    #     o3d.visualization.draw_geometries([before_pcd, after_pcd], window_name="Before and After Point Clouds")
    transformation, fitness, rmse = run_icp(after_pcd, before_pcd, args.threshold)
    # print(f"Fitness: {fitness:.4f}, RMSE: {rmse:.4f}")
    after_pcd.transform(transformation)
    if args.visualize:
        o3d.visualization.draw_geometries([before_pcd, after_pcd], window_name="ICP Result")

    time_end1 = time.time()
    print(f"ICP Point2Plane After → Before \t completed in {time_end1 - time_start1:.2f} seconds")
    time_start2 = time.time()
    before_points = np.asarray(before_pcd.points)
    after_points = np.asarray(after_pcd.points)

    after_mask = (after_points[:, 0] > 0) & (after_points[:, 0] < args.roi_radius) & (after_points[:, 1] > 0) & (after_points[:, 1] < args.roi_radius) & (after_points[:, 2] > 0) & (after_points[:, 2] < args.roi_radius)
    after_roi_candidate_points = after_points[after_mask]

    # Create point clouds for ROI
    after_roi_pcd = o3d.geometry.PointCloud()
    after_roi_pcd.points = o3d.utility.Vector3dVector(after_roi_candidate_points)

    after_clustering = DBSCAN(eps=args.cluster_eps, min_samples=args.cluster_min_points).fit(after_roi_candidate_points)
    after_labels = after_clustering.labels_
    after_unique_labels = set(after_labels)
    after_min_dist = float('inf')
    after_best_label = -1
    for label in after_unique_labels:
        if label == -1:
            continue
        cluster_points = after_roi_candidate_points[after_labels == label]
        center = cluster_points.mean(axis=0)
        dist = np.linalg.norm(center)
        if dist < after_min_dist:  
            after_min_dist = dist
            after_best_label = label
    
    # best_label이 가장 원점에 가까운 cluster
    before_roi_points = np.load(args.input_before_roi)
    after_roi_points = after_roi_candidate_points[after_labels == after_best_label]
    np.save(args.output_after_roi, after_roi_points)
    
    if args.visualize:
        before_roi_pcd = o3d.geometry.PointCloud()
        before_roi_pcd.points = o3d.utility.Vector3dVector(before_roi_points)
        before_roi_pcd.paint_uniform_color([0.3, 0.3, 1.0])
        after_roi_pcd.points = o3d.utility.Vector3dVector(after_roi_points)
        after_roi_pcd.paint_uniform_color([1.0, 0.3, 0.3])
        o3d.visualization.draw_geometries([before_roi_pcd, after_roi_pcd], window_name="Region of Interest")

    
    time_end2 = time.time()
    print(f"Extract Region of Interest \t completed in {time_end2 - time_start2:.2f} seconds")
    time_start3 = time.time()

    # Calculate volume difference
    x_min, y_min = 0, 0
    x_max, y_max = np.max(np.concatenate((before_roi_points[:, 0], after_roi_points[:, 0]), axis=0)), np.max(np.concatenate((before_roi_points[:, 1], after_roi_points[:, 1]), axis=0))

    before_grid_z = defaultdict(list)
    after_grid_z = defaultdict(list)
    for point in before_roi_points:
        x_idx = int(point[0] / args.grid_size)
        y_idx = int(point[1] / args.grid_size)
        if 0 <= x_idx < int((x_max - x_min) / args.grid_size) and 0 <= y_idx < int((y_max - y_min) / args.grid_size):
            before_grid_z[(x_idx, y_idx)].append(point[2])
    for point in after_roi_points:
        x_idx = int(point[0] / args.grid_size)
        y_idx = int(point[1] / args.grid_size)
        if 0 <= x_idx < int((x_max - x_min) / args.grid_size) and 0 <= y_idx < int((y_max - y_min) / args.grid_size):
            after_grid_z[(x_idx, y_idx)].append(point[2])
    before_grid_z_medians = {k: np.median(v) for k, v in before_grid_z.items() if len(v) > 0}
    after_grid_z_medians = {k: np.median(v) for k, v in after_grid_z.items() if len(v) > 0}
    
    # if args.visualize:
    #     before_grid_z_cloud = o3d.geometry.PointCloud()
    #     before_grid_z_cloud.points = o3d.utility.Vector3dVector(
    #         np.array([[k[0] * args.grid_size, k[1] * args.grid_size, v] for k, v in before_grid_z_medians.items()])
    #     )
    #     before_grid_z_cloud.paint_uniform_color([0.3, 0.3, 1.0])
    #     after_grid_z_cloud = o3d.geometry.PointCloud()
    #     after_grid_z_cloud.points = o3d.utility.Vector3dVector(
    #         np.array([[k[0] * args.grid_size, k[1] * args.grid_size, v] for k, v in after_grid_z_medians.items()])
    #     )
    #     after_grid_z_cloud.paint_uniform_color([1.0, 0.3, 0.3])
    #     o3d.visualization.draw_geometries([before_grid_z_cloud, after_grid_z_cloud], window_name="Grid Z Medians")

    x_size = int((x_max - x_min) / args.grid_size)
    y_size = int((y_max - y_min) / args.grid_size)
    z_diff = np.zeros((x_size, y_size))
    for (i, j) in before_grid_z_medians:
        if (i, j) in after_grid_z_medians:
            z_diff[i, j] = before_grid_z_medians[(i, j)] - after_grid_z_medians[(i, j)]

    volume_diff = np.sum(z_diff) * (args.grid_size ** 2)


    if args.visualize:
        z_diff_cloud = o3d.geometry.PointCloud()
        z_diff_points = np.array([[i * args.grid_size, j * args.grid_size, z_diff[i, j]] for i in range(x_size) for j in range(y_size)])
        z_diff_cloud.points = o3d.utility.Vector3dVector(z_diff_points)
        z_diff_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([z_diff_cloud], window_name="Z Difference Cloud")

    time_end3 = time.time()
    print(f"Calculate volume difference \t completed in {time_end3 - time_start3:.2f} seconds")
    print("\n=============== Volume Calculation Summary ===============\n")
    print(f"\tVolume difference (Before - After) = {volume_diff:.2f} m³\n")
    print("==========================================================\n")

if __name__ == "__main__":
    main()

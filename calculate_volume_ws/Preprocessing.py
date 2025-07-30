#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import rosbag2_py
import struct
import argparse
import time
from rclpy.serialization import deserialize_message
from sklearn.cluster import DBSCAN
from rosidl_runtime_py.utilities import get_message

def process_lidar_frames(bag_dir, topic_name="/livox/lidar", num_frames=10, voxel_size=0.05, visualize=False, output_file="output.pcd"):

    # Setup ROS2 bag reader
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # Get topic types
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if topic_name not in type_map:
        print(f"Topic '{topic_name}' not found in bag.")
        return
    
    msg_type = get_message(type_map[topic_name])
    frame_count = 0
    all_points = []

    while reader.has_next() and frame_count < num_frames:
        topic, data, _ = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)

        point_step = msg.point_step
        if point_step < 12:
            continue  # x, y, z each float32 = 12 bytes minimum
    
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, point_step)
        xyz = np.frombuffer(raw[:, :12].flatten(), dtype=np.float32).reshape(-1, 3)

        # Filtering invalid or far-away points
        valid_mask = np.all(np.abs(xyz) < 1000, axis=1) & ~np.isnan(xyz).any(axis=1)
        xyz = xyz[valid_mask]

        if xyz.shape[0] > 0:
            all_points.append(xyz)
            frame_count += 1

    if len(all_points) == 0:
        print("No valid points found in the specified frames.")
        return
    
    # Combine all points into a single point cloud
    all_points = np.vstack(all_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Remove outliers using statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_pcd = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(output_file, inlier_pcd)
    
    if visualize:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([inlier_pcd, coord_frame])

    return inlier_pcd

def extract_planes_and_reference(
    preprocessed_pcd,
    output_ref_path,
    ransac_dist,
    ransac_iter,
    num_planes,
    perp_threshold,
    connection_threshold,
    min_intersection_points,
    plane_offset,
    visualize=False,
):
    # Load point cloud
    pcd = preprocessed_pcd
    planes_cloud = []
    planes_model = []
    extract_radius = 40.0
    points = np.asarray(pcd.points)
    distance = np.linalg.norm(points, axis=1)
    indices = np.where(distance < extract_radius)[0]
    rest_cloud = pcd.select_by_index(indices)

    # Plane extraction loop
    while True:
        rest_cloud = pcd.select_by_index(indices)
        base_planes_model = []
        base_planes_cloud = []
        break_flag = False
        for _ in range(num_planes):
            if len(rest_cloud.points) <= 1000 or len(base_planes_model) >= 3:
                break
            # RANSAC plane segmentation
            plane_model, inliers = rest_cloud.segment_plane(
                distance_threshold=ransac_dist,
                ransac_n=3,
                num_iterations=ransac_iter
            )
            if len(inliers) < 1000:
                break
            
            [a, b, c, d] = plane_model
            normal = np.array([a, b, c])
            plane_model /= np.linalg.norm(normal)  # Normalize the normal vector
            if plane_model[3] < 0:  # Ensure consistent orientation
                plane_model = -plane_model
            plane_cloud = rest_cloud.select_by_index(inliers)

            if base_planes_model == []:
                if np.abs(normal[2]) > 0.9:
                    base_planes_model.append(plane_model)
                    plane_cloud.paint_uniform_color([0.0, 0.0, 1.0])  
                    base_planes_cloud.append(plane_cloud)
                else:
                    continue
            else:
                normal_b = np.array(base_planes_model[0][:3])
                if np.abs(np.dot(normal, normal_b)) < perp_threshold:
                    planes_model.append(plane_model)
                    planes_cloud.append(plane_cloud)
                    if len(planes_model) > 2:
                        for i in range(len(planes_model)-1):
                            dist = planes_cloud[i].compute_point_cloud_distance(plane_cloud)
                            if np.abs(np.dot(normal, np.array(planes_model[i][:3]))) < perp_threshold and np.sum(np.array(dist) < connection_threshold) > min_intersection_points:
                                planes_cloud[i].paint_uniform_color([1.0, 0.0, 0.0])  # Paint existing plane
                                base_planes_cloud.append(planes_cloud[i])
                                plane_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Paint existing plane
                                base_planes_cloud.append(plane_cloud)
                                base_planes_model.append(planes_model[i])
                                base_planes_model.append(plane_model)
                                break_flag = True
                                break
                    areas = [len(pc.points) for pc in planes_cloud]
                    sorted_indices = np.argsort(areas)[::-1]
                    planes_cloud = [planes_cloud[i] for i in sorted_indices]
                    planes_model = [planes_model[i] for i in sorted_indices]
            rest_cloud = rest_cloud.select_by_index(inliers, invert=True)  # 나머지 포인트클라우드 업데이트
            if break_flag:
                break
        if break_flag:
            break

    # Frame extraction
    def normalize(v):
        return v / np.linalg.norm(v)
    
    z_axis = np.array(base_planes_model[0][:3])
    if np.dot(np.cross(base_planes_model[1][:3], z_axis), base_planes_model[2][:3]) < 0:
        x_axis = np.array(base_planes_model[1][:3])
    else:
        x_axis = np.array(base_planes_model[2][:3])
    x_axis = normalize(x_axis - np.dot(x_axis, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)

    A = np.array([base_planes_model[0][:3], base_planes_model[1][:3], base_planes_model[2][:3]])
    d = np.array([-base_planes_model[0][3], -base_planes_model[1][3], -base_planes_model[2][3]])
    
    origin = np.linalg.solve(A, d)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    origin = origin + np.array([plane_offset, plane_offset, plane_offset]) @ R.T
    # SE(3) 4x4 변환행렬 생성
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin
    np.save(output_ref_path, T)

    if visualize:
        def make_axis_cloud(origin, axis, color, length=2.0, num_points=100):
            points = [origin + axis * (length * t / num_points) for t in range(num_points)]
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.paint_uniform_color(color)
            return pc
        x_pc = make_axis_cloud(origin, x_axis, [1,0,0], length=2.0)
        y_pc = make_axis_cloud(origin, y_axis, [0,1,0], length=2.0)
        z_pc = make_axis_cloud(origin, z_axis, [0,0,1], length=2.0)    
        pcd_copy = pcd
        pcd_copy.paint_uniform_color([0.9, 0.9, 0.9])
        o3d.visualization.draw_geometries(
            [pcd_copy] + base_planes_cloud + [x_pc, y_pc, z_pc],
            window_name="Extracted Planes & Reference Axes"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input and output Paths
    parser.add_argument("--input", type=str, required=True, help="Input bag directory")
    parser.add_argument("--output-pcd", type=str, required=True, help="Output preprocessed PCD file (.pcd)")
    parser.add_argument("--output-ref", type=str, required=True, help="Output reference frame (.npy)")
    
    # Preprocessing parameters
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to combine")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for downsampling")

    # RANSAC parameters
    parser.add_argument("--ransac-dist", type=float, default=0.05,help="RANSAC distance threshold")
    parser.add_argument("--ransac-iter", type=int, default=1000, help="RANSAC max iterations")
    parser.add_argument("--num-planes", type=int, default=10, help="RANSAC max number of planes to extract")

    # Frame extraction parameters
    parser.add_argument("--perp-threshold", type=float, default=0.1, help="Perpendicularity threshold (dot product)")
    parser.add_argument("--connection-threshold", type=float, default=0.3, help="Connection threshold (intersection)")
    parser.add_argument("--min-intersection-points", type=int, default=10, help="Minimum points for plane intersection")
    parser.add_argument("--plane-offset", type=float, default=0.1, help="Offset for plane extraction")
    
    # Visualization flag
    parser.add_argument("--visualize", action="store_true", help="Visualize point cloud")
    
    args = parser.parse_args()

    start_time1 = time.time()
    preprocessed_pcd = process_lidar_frames(
        bag_dir=args.input,
        topic_name="/livox/lidar",
        num_frames=args.frames,
        voxel_size=args.voxel_size,
        visualize=args.visualize,
        output_file=args.output_pcd
    )
    end_time1 = time.time()
    print(f"Convert ROS2 bag to Pointcloud \t completed in {end_time1 - start_time1:.2f} seconds")

    start_time2 = time.time()
    extract_planes_and_reference(
        preprocessed_pcd=preprocessed_pcd,
        output_ref_path=args.output_ref,
        ransac_dist=args.ransac_dist,
        ransac_iter=args.ransac_iter,
        num_planes=args.num_planes,
        perp_threshold=args.perp_threshold,
        connection_threshold=args.connection_threshold,
        min_intersection_points=args.min_intersection_points,
        plane_offset=args.plane_offset,
        visualize=args.visualize,
    )
    end_time2 = time.time()
    print(f"Extract reference frame \t completed in {end_time2 - start_time2:.2f} seconds")
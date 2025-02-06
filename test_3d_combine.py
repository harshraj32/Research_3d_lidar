import open3d as o3d
import numpy as np
import json

def load_calib_transformation(calib_json_path):
    """Load the calibration transformation matrix from JSON."""
    with open(calib_json_path, 'r') as f:
        calib = json.load(f)
    
    rotation = np.array(calib["rotation"])
    translation = np.array(calib["translation"])
    
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation.flatten()
    
    return transformation

def load_point_cloud(pcd_path, color):
    """Load a PCD file and apply a uniform color."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        print(f"Warning: {pcd_path} is empty or could not be loaded.")
    else:
        pcd.paint_uniform_color(color)  # Apply color
    return pcd

def save_and_visualize_point_clouds(vehicle_pcd, infra_pcd, transformation, save_prefix):
    """Apply transformation to infrastructure PCD, save and visualize."""
    
    # Save the original vehicle and infrastructure point clouds
    o3d.io.write_point_cloud(f"{save_prefix}_vehicle.pcd", vehicle_pcd)
    o3d.io.write_point_cloud(f"{save_prefix}_infrastructure.pcd", infra_pcd)
    
    # Apply initial transformation to the infrastructure PCD
    infra_pcd_initial = infra_pcd.transform(transformation)
    
    # Save initial alignment
    o3d.io.write_point_cloud(f"{save_prefix}_initial_alignment.pcd", vehicle_pcd + infra_pcd_initial)
    print(f"Saved initial alignment to {save_prefix}_initial_alignment.pcd")

    # Visualize initial alignment
    o3d.visualization.draw_geometries([vehicle_pcd, infra_pcd_initial],
                                      window_name="Initial Alignment")

    # Perform ICP registration to refine the transformation
    threshold = 1.0  # Adjust based on your data
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=infra_pcd_initial, target=vehicle_pcd, max_correspondence_distance=threshold,
        init=np.eye(4),  
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print("ICP Refinement Transformation:")
    print(reg_p2p.transformation)

    # Apply refined transformation
    infra_pcd_final = infra_pcd_initial.transform(reg_p2p.transformation)

    # Save final alignment
    o3d.io.write_point_cloud(f"{save_prefix}_final_alignment.pcd", vehicle_pcd + infra_pcd_final)
    print(f"Saved final alignment to {save_prefix}_final_alignment.pcd")

    # Visualize refined alignment
    o3d.visualization.draw_geometries([vehicle_pcd, infra_pcd_final],
                                      window_name="Refined Alignment")

    # Merge point clouds and save
    combined_pcd = vehicle_pcd + infra_pcd_final
    o3d.io.write_point_cloud(f"{save_prefix}_combined.pcd", combined_pcd)
    print(f"Saved combined point cloud to {save_prefix}_combined.pcd")

    # Visualize final combined point cloud
    o3d.visualization.draw_geometries([combined_pcd], window_name="Final Combined Point Cloud")

def main():
    """Main function to process and visualize the LiDAR data."""
    
    # Paths based on your dataset structure
    vehicle_pcd_path = "vehicle-side/velodyne/015376.pcd"
    infra_pcd_path = "infrastructure-side/velodyne/000020.pcd"
    calib_path = "cooperative/calib/lidar_i2v/015376.json"

    # Load the point clouds with assigned colors
    vehicle_pcd = load_point_cloud(vehicle_pcd_path, [0.0, 1.0, 0.0])  # Green for vehicle
    infra_pcd = load_point_cloud(infra_pcd_path, [1.0, 0.0, 0.0])  # Red for infrastructure

    # Load transformation from calibration
    transformation = load_calib_transformation(calib_path)

    # Save and visualize with transformation and ICP refinement
    save_and_visualize_point_clouds(vehicle_pcd, infra_pcd, transformation, "output")

if __name__ == "__main__":
    main()


def main():
    """Main function to process and visualize the LiDAR data."""
    
    # Paths based on your dataset structure
    vehicle_pcd_path = "vehicle-side/velodyne/015376.pcd"
    infra_pcd_path = "infrastructure-side/velodyne/000020.pcd"
    calib_path = "cooperative/calib/lidar_i2v/015376.json"
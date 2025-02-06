import numpy as np
import open3d as o3d
import json
import yaml
from jgmm import jgmm

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_point_cloud(pcd_path):
    return o3d.io.read_point_cloud(pcd_path)

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

def apply_system_error_offset(transform, delta_x, delta_y):
    offset_matrix = np.eye(4)
    offset_matrix[0, 3] = delta_x
    offset_matrix[1, 3] = delta_y
    return np.dot(transform, offset_matrix)

def icp_registration(source, target, threshold=0.02, max_iteration=50):
    icp = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    return icp.transformation

def compute_alignment_loss(computed_matrix, ground_truth_matrix):
    return np.linalg.norm(computed_matrix - ground_truth_matrix, ord='fro')

def process_pair_icp(vehicle_pcd, infra_pcd, gt_matrix):
    """Process a single pair using ICP and return the loss."""
    transform = icp_registration(vehicle_pcd, infra_pcd)
    return compute_alignment_loss(transform, gt_matrix)

def create_init_pc(min_bound, max_bound, num_points=400):
    """Create initial point cloud within specified bounds"""
    # Create points in a box shape within the bounds
    x = np.random.uniform(min_bound[0], max_bound[0], num_points)
    y = np.random.uniform(min_bound[1], max_bound[1], num_points)
    z = np.random.uniform(min_bound[2], max_bound[2], num_points)
    
    points = np.vstack((x, y, z))  # Shape: (3, num_points)
    return points
def process_pair_jgmm(vehicle_pcd, infra_pcd, gt_matrix, config):
    """Process a single pair using JGMM and return the loss."""
    try:
        # Get bounds from config
        min_bound = np.array(config.get('min_bound', [-50, -50, -50]))
        max_bound = np.array(config.get('max_bound', [50, 50, 50]))
        
        # Convert point clouds to numpy arrays (Ensure (N, 3) shape)
        vehicle_points = np.asarray(vehicle_pcd.points)  # Shape: (N, 3)
        infra_points = np.asarray(infra_pcd.points)      # Shape: (N, 3)
        
        # Create list of point clouds for JGMM
        V = [vehicle_points, infra_points]
        
        # Create initial centroids and ensure correct shape
        Xin = create_init_pc(min_bound, max_bound).T  # Shape: (3, num_points)
        
        # Run JGMM
        X, TV, T, pk = jgmm(V, Xin, maxNumIter=100)
        
        # Get the final transformation (for infra relative to vehicle)
        final_transform = np.eye(4)
        final_transform[:3, :3] = T[-1][0][1]   # Rotation (OK)
        final_transform[:3, 3] = T[-1][1][1].reshape(3)  # Ensure translation shape is (3,)

        
        return compute_alignment_loss(final_transform, gt_matrix)
    
    except Exception as e:
        print(f"JGMM processing failed: {str(e)}")
        return float('inf')


# Main execution
def main():
    # Load configuration
    try:
        config = load_config('config/config.yaml')
    except FileNotFoundError:
        print("Config file not found, using default values")
        config = {
            'min_bound': [-50, -50, -50],
            'max_bound': [50, 50, 50]
        }

    # Load the data
    try:
        with open('cooperative/data_info_new.json', 'r') as f:
            data_info = json.load(f)
    except FileNotFoundError:
        print("Error: data_info_new.json not found")
        return

    # Process all pairs and compute losses
    icp_losses = []
    jgmm_losses = []

    for idx, pair in enumerate(data_info):
        try:
            print(f"\nProcessing pair {idx + 1}/{len(data_info)}")
            
            # Load point clouds
            vehicle_pcd = o3d.io.read_point_cloud(pair["vehicle_pointcloud_path"])
            infra_pcd = o3d.io.read_point_cloud(pair["infrastructure_pointcloud_path"])
            
            # Load ground truth calibration matrix
            gt_matrix = load_calib_transformation(pair["calib_lidar_i2v_path"])
            
            # Apply system error offset
            gt_matrix = apply_system_error_offset(gt_matrix, 
                                                pair["system_error_offset"]["delta_x"],
                                                pair["system_error_offset"]["delta_y"])
            
            # Process using ICP
            icp_loss = process_pair_icp(vehicle_pcd, infra_pcd, gt_matrix)
            icp_losses.append(icp_loss)
            print(f"ICP Loss: {icp_loss:.4f}")
            
            # Process using JGMM
            jgmm_loss = process_pair_jgmm(vehicle_pcd, infra_pcd, gt_matrix, config)
            if jgmm_loss != float('inf'):
                jgmm_losses.append(jgmm_loss)
                print(f"JGMM Loss: {jgmm_loss:.4f}")
            else:
                print("JGMM processing failed for this pair")
                
        except Exception as e:
            print(f"Error processing pair {idx + 1}: {str(e)}")
            continue

    # Calculate and print average losses
    if icp_losses:
        avg_icp_loss = np.mean(icp_losses)
        print(f"\nAverage ICP Loss: {avg_icp_loss:.4f}")
        print(f"Number of successful ICP registrations: {len(icp_losses)}")

    if jgmm_losses:
        avg_jgmm_loss = np.mean(jgmm_losses)
        print(f"Average JGMM Loss: {avg_jgmm_loss:.4f}")
        print(f"Number of successful JGMM registrations: {len(jgmm_losses)}")

if __name__ == "__main__":
    main()
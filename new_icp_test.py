import open3d as o3d
import copy
import numpy as np

# Initialize functions
def draw_registration_result(source, target, transformation):
    """
    Visualize the registration result.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])

def find_nearest_neighbors(source_pc, target_pc, nearest_neigh_num):
    """
    Find the closest neighbor for each point using KDTree.
    """
    point_cloud_tree = o3d.geometry.KDTreeFlann(source_pc)
    points_arr = []
    for point in target_pc.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, nearest_neigh_num)
        points_arr.append(source_pc.points[idx[0]])
    return np.asarray(points_arr)

def icp(source, target):
    """
    Perform ICP registration between source and target point clouds.
    Returns the final transformation matrix, rotation matrix, and translation vector.
    """
    source.paint_uniform_color([0.5, 0.5, 0.5])
    target.paint_uniform_color([0, 0, 1])
    
    target_points = np.asarray(target.points)

    # Initialize transformation matrix
    transform_matrix = np.asarray([
        [0.862, 0.011, -0.507, 0.5], 
        [-0.139, 0.967, -0.215, 0.7], 
        [0.487, 0.255, 0.835, -1.4], 
        [0.0, 0.0, 0.0, 1.0]
    ])
    source.transform(transform_matrix)

    # While loop variables
    curr_iteration = 0
    cost_change_threshold = 0.001
    curr_cost = 1000
    prev_cost = 10000

    while True:
        # 1. Find nearest neighbors
        new_source_points = find_nearest_neighbors(source, target, 1)

        # 2. Compute centroids and reposition points
        source_centroid = np.mean(new_source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_repos = new_source_points - source_centroid
        target_repos = target_points - target_centroid

        # 3. Compute covariance matrix and SVD
        cov_mat = target_repos.T @ source_repos
        U, _, Vt = np.linalg.svd(cov_mat)
        R = U @ Vt

        # Ensure R is a proper rotation matrix (no reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
            print("Reflection detected in SVD, fixing rotation matrix.")

        t = target_centroid - R @ source_centroid
        t = t.reshape((3, 1))

        # Compute cost change
        curr_cost = np.linalg.norm(target_repos - (R @ source_repos.T).T)
        print("Curr_cost=", curr_cost)

        if (prev_cost - curr_cost) > cost_change_threshold:
            prev_cost = curr_cost
            transform_matrix = np.hstack((R, t))
            transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))

            # Apply transformation
            source.transform(transform_matrix)
            curr_iteration += 1
        else:
            break

    print("\nFinal Iteration Count =", curr_iteration)
    
    # Extract final rotation and translation
    final_rotation = transform_matrix[:3, :3]
    final_translation = transform_matrix[:3, 3]

    print("\nFinal Rotation Matrix:")
    print(final_rotation)

    print("\nFinal Translation Vector:")
    print(final_translation)

    # Visualize final result
    draw_registration_result(source, target, transform_matrix)
    
    return transform_matrix, final_rotation, final_translation


### **Load Point Clouds** ###
target = o3d.io.read_point_cloud("infrastructure-side/velodyne/000020.pcd")
source = o3d.io.read_point_cloud("vehicle-side/velodyne/015376.pcd")

print("Loaded Source Points:", len(source.points))
print("Loaded Target Points:", len(target.points))

# Run ICP
final_transform, estimated_rotation, estimated_translation = icp(source, target)

# Apply final transformation before saving
source.transform(final_transform)

# Save the final transformed source point cloud
o3d.io.write_point_cloud("transformed_source.pcd", source)
print("Final transformed point cloud saved as 'transformed_source.pcd'")

# Given transformation for comparison
given_rotation = np.array([
    [-0.4545614699109483, -0.8907153435001919, 0.0005648951323945557],
    [0.8907146149767075, -0.45456011348533937, 0.0016613333577826374],
    [-0.0012230438466124427, 0.0012583389713516793, 0.9999986684102067]
])
given_translation = np.array([39.616827594133596, -17.046397643081917, -0.03697431312528282])

# Compute differences
rotation_difference = np.linalg.norm(given_rotation - estimated_rotation)
translation_difference = np.linalg.norm(given_translation - estimated_translation)

# Print final results
print("\nRotation Difference:", rotation_difference)
print("Translation Difference:", translation_difference)

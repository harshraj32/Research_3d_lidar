import argparse
import os

import numpy as np
import pickle

def calibrate(data_path, config_file_path, sequence):
    pcds = generate_data(data_path, config_file_path, sequence)
    Xin = create_init_pc(box_size=(0.5, 0.5, 0.5), num_points=400) + np.array([9.8, 4.75, 0.38])

    V = [np.array(cloud.points) for cloud in pcds]
    nObs = len(V)

    print("####### Perform Calibration and Model Generation. ########")
    X, TV, AllT, pk= jgmm(V=V, Xin=Xin, maxNumIter=100)

 
    T_1 = [homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2)]
    T_2 = [homogeneous_transform(AllT[-1][0][i], AllT[-1][1][i].reshape(-1)) for i in range(nObs // 2, nObs)]

    T_calib = [np.dot(np.linalg.inv(T_2[i]), T_1[i]) for i in range(len(T_1))]
    T_final = mean_transform(T_calib)
    print("Calibration Error: \n")
    print(T_final)
    gmmcalib_result = [T_final, X]
    with open("/app/output/gmmcalib_result.pkl", "wb") as f:
        pickle.dump(gmmcalib_result, f) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration script")
    parser.add_argument("--data_path", type=str, help="Path to data", default="../data")
    parser.add_argument("--config_file_path", type=str, help="Path to config file", default="../config/config.yaml")
    parser.add_argument("--sequence", nargs='+', type=int, help="Sequence sequence of pcds")

    args = parser.parse_args()

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_path))
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), args.config_file_path))

    if args.sequence is None:
        sequence = list(range(1, len(os.listdir(str(data_path+"/sensor_1"))) + 1))
    else:
        sequence = args.sequence

    calibrate(data_path, config_file_path, sequence)



import open3d as o3d
import numpy as np
###################################################################################
#########           M O D E L    G E N E R A T I O N     ##########################
###################################################################################

"""
This code implementation was inpired by G. D. Evangelidis and R. Horaud, 
“Joint Alignment of Multiple Point Sets with Batch and Incremental Expectation-Maximization,” 
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, pp. 1397–1410, June 2018.
"""

def jgmm(V, Xin, maxNumIter):
    """Calculate the transformations and jointly align points clouds
    Parameters
    ---------------
    V: list, M
        containing the measurement point clouds
    Xin: (dim, K) np.array
        initial GMM centroids as Point Cloud
    Returns
    ---------------
    X: array, shape (N, 3)
        jgmm model point cloud
    TV: list, shape (M)
        Transformed views
    T: list,
        Transformations at each iteration
    pk: array, shape(K, 1)
        probability of each point in the generated model

    """

    # -------------------------------------------------------------------- #
    #               Initialize Variables and Matrices                      #
    # -------------------------------------------------------------------- #

    V = [np.transpose(i) for i in V]
    X = np.transpose(Xin)
    TV = [] 

    """Number of Measurments"""
    M = len(V)
    """Number of Centroids """
    dim, K = X.shape

    """Init rotation matrix"""
    R = []
    for i in range(M):
        R.append(np.eye(3))

    """ Init translation matrix"""
    t = []
    for i in range(len(V)):
        t.append(np.array([0, 0, 0]))

    """ Transformed Sets based on initial R & t"""
    TV = [np.dot(R[i],V[i]) + t[i].reshape((3,1)) for i in range(len(V))]

    """ Initial Covariances for the centroids"""
    minXYZ, maxXYZ = [], []
    TVX = TV.copy()
    TVX.append(X)
    for i in range(len(TVX)):
        minXYZ.append(np.min(TVX[i], axis = 1))
        maxXYZ.append(np.max(TVX[i], axis = 1))

    minXYZ = np.min(minXYZ, axis=0).reshape((dim,1))
    maxXYZ = np.max(maxXYZ, axis=0).reshape((dim,1))

    Q = np.multiply(np.ones((1, K)), (1/ sse(minXYZ, maxXYZ) ) ).reshape((K,1)).astype(np.float64)

    #maxNumIter = 10
    epsilon = 1e-9
    updatePriors = 1
    gamma =  0.1
    pk = 1/(K*(gamma+1))


    # -------------------------------------------------------------------- #
    #               E    M    A L G O R I T H M                            #
    # -------------------------------------------------------------------- #

    h = np.divide(2, np.mean(Q))
    beta = np.divide(gamma, np.multiply(h, gamma+1))
    pk = np.transpose(pk)
    T = []

    for it in range(maxNumIter):
        print("GMM Iteration: ", it)
        ''' Calculate Posteriors '''
        ''' Squared Error Between transformed frames and compontents'''
        alpha = [sse(np.asarray(i), np.asarray(X)) for i in TV]
        ''' Correspondences '''
        alpha = [np.multiply(np.multiply(pk, np.transpose(Q) ** 1.5),np.exp(np.multiply(-0.5 * np.transpose(Q), i))) for i in alpha]
        '''Normalization with the sum of alpha and beta'''
        alpha = [np.divide(i.T, np.asmatrix(np.sum(i, axis=1) + beta)).T for i in alpha]

        '''Weights '''
        lmda = [np.sum(i, axis= 0).T  for i in alpha]

        W = [np.multiply(np.dot(V[i], alpha[i]), Q.T) for i in range(len(V))]

        b = [np.multiply(i, Q) for i in lmda]

        '''mean of W'''
        mW = [np.sum(i, axis= 1) for i in W]

        '''mean of X'''
        mX = [np.dot(X, i) for i in b]

        sumOfWeights = [i.T.dot(Q)[0,0] for i in lmda]

        P = [np.dot(X, W[i].T)- (np.dot(mX[i], mW[i].T)/sumOfWeights[i])  for i in range(len(W))]

        '''SVD'''
        uu, ss, vv = [], [],[]
        for i in range(len(P)):
            u, s, v = np.linalg.svd(P[i])
            uu.append(u)
            ss.append(s)
            vv.append(v)


        '''Find optimal rotation'''

        R = [np.dot( uu[i].dot(np.diag([1, 1, np.linalg.det(np.dot(uu[i], vv[i].T))])),vv[i]) for i in range(len(uu))]

        ''' Find optimal translation'''
        t = [(mX[i] - R[i].dot(mW[i])) / sumOfWeights[i] for i in range(len(R))]


        '''Populate T'''
        T.append((R, t))

        '''Transformed Sets'''
        TV = [R[i].dot(V[i]) + t[i] for i in range(len(R))]

        '''Update X'''
        lmdaMatrix = np.asarray(lmda).astype(np.float64)
        den = np.sum(np.moveaxis(lmdaMatrix, 0, 1), axis=1).T

        X = [TV[i].dot(alpha[i]) for i in range(len(TV))] # (M, 3, K) Matrix
        X = np.sum(np.stack(np.asarray(X[:]), axis=0), axis=0)
        X = X/den

        '''Update Covariances '''
        wnormes = [np.sum(np.multiply(alpha[i], sse(np.asarray(TV[i].astype(np.float64)), np.asarray(X))), axis=0) for i in range(len(TV))]

        Q = np.transpose(np.divide(3*den, np.sum(np.stack(np.asarray(wnormes[:]), axis=0), axis=0) + 3*den*epsilon))

        if updatePriors:
            pk = den / ((gamma+1)*sum(den))

    Q = np.divide(1, Q)
    return X, TV, T, pk


def sse(A, B):
    """Compute the Sum Of Squared Error of two matrices.
        Returns
        -------
        C : list, shape (N_train, 4)
            SSE of the two matrices.
        """

    A = np.moveaxis(A[np.newaxis, :, :], 0, -1) # results in a (3, N, 1) matrix
    B = np.swapaxes(np.moveaxis(B[np.newaxis, :, :], 0, -1), 1, -1) # results in a (3, 1, K) matrix

    C = np.sum(np.power((A - B), 2), axis=0) # sum over the the first axis of the A and B (three dimensions)
    if isinstance(C, (list, tuple, np.ndarray)):
        return C
    else:
        return C[0][0]
    



import numpy as np
from scipy.spatial.transform import Rotation as R 

def compute_global_transform(xyz_cal, rpy_cal):

    rotation_matrix = R.from_euler('xyz', rpy_cal, degrees=False).as_matrix()
    
    calibration_matrix = np.eye(4)
    calibration_matrix[:3, :3] = rotation_matrix
    calibration_matrix[:3, 3] = xyz_cal

    return calibration_matrix

def homogeneous_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def homogeneous_to_euler(T):
    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if np.abs(pitch - np.pi / 2) < 1e-6:
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    elif np.abs(pitch + np.pi / 2) < 1e-6:
        # Gimbal lock case
        roll = 0
        yaw = -np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    euler_angles = np.array([roll, pitch, yaw])
    return euler_angles

def euler_to_homogeneous(roll_deg, pitch_deg, yaw_deg, translation):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    R = np.array([[cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
                  [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
                  [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def mean_transform(T):
    
    euler_list = [homogeneous_to_euler(i) for i in T]
    translation_list = [i[:3,3] for i in T]

    euler_mean = np.mean(euler_list, axis=0)
    translation_mean = np.mean(translation_list, axis=0)
    
    return euler_to_homogeneous(np.rad2deg(euler_mean)[0], np.rad2deg(euler_mean)[1], np.rad2deg(euler_mean)[2], translation_mean)


import numpy as np
import open3d as o3d
import pickle
import yaml


def generate_data(data_path, config_file_path, sequence):
    # Read the parameters from the YAML file
    with open(config_file_path, 'r') as file:
        config_data = yaml.safe_load(file)

    transform_sensor_1 = config_data.get("transform_sensor_1", "")[0]
    transform_sensor_2 = config_data.get("transform_sensor_2", "")[0]
    min_bound = config_data.get("min_bound", "")[0]
    max_bound = config_data.get("max_bound", "")[0]
    number_of_sensors = config_data.get("number_of_sensors", "")
    
    sensors = [data_path + "/sensor_" + str(i+1) + "/" for i in range(number_of_sensors)]

    pcds = []
    for sensor in sensors:
        for frame in range(sequence[0], sequence[-1]+1):
            pcd = o3d.io.read_point_cloud(sensor + str(frame) + ".pcd")

            if sensor == sensors[0]:
                T_g = compute_global_transform(transform_sensor_1[3:], transform_sensor_1[:3])
            else: 
                T_g = compute_global_transform(transform_sensor_2[3:], transform_sensor_2[:3])

            pcd.transform(T_g)
            # Crop 
            roi = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcds.append(pcd.crop(roi))
    return pcds



def create_init_pc(box_size, num_points):
    # Create a box mesh
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])

    # Sample points on each face of the box
    sampled_points = []

    for face in box_mesh.triangles:
        # Get the vertices of the face
        vertices = np.asarray(box_mesh.vertices)[face]

        # Compute the normal of the face
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal /= np.linalg.norm(normal)
        np.random.seed(3)
        # Sample points on the face using the barycentric coordinates method
        u = np.random.rand(num_points//6)
        v = np.random.rand(num_points//6)
        mask = u + v < 1
        u = u[mask]
        v = v[mask]
        points_on_face = vertices[0] + u[:, None] * (vertices[1] - vertices[0]) + v[:, None] * (vertices[2] - vertices[0])

        # Project points onto the plane of the face
        points_on_plane = points_on_face - np.dot(points_on_face - vertices[0], normal)[:, None] * normal

        # Add the sampled points to the list
        sampled_points.extend(points_on_plane)

    return np.array(sampled_points)


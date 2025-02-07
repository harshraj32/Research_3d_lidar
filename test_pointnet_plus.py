import torch
import torch.nn as nn
import torch.optim as optim
import open3d as o3d
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import penalties as P  

# Import user-provided PointNet++ architecture
from pointnetplus_model import PointNet  

# Check for MPS (Metal) support on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
class PointCloudDataset(Dataset):
    def __init__(self, json_path, n_points=2048):
        """
        Args:
            json_path: Path to JSON file containing dataset information
            n_points: Number of points to sample from each point cloud
        """
        with open(json_path, "r") as f:
            self.data_info = json.load(f)
        self.n_points = n_points

    def random_sample_points(self, points, n_points):
        """Randomly sample a fixed number of points from a point cloud."""
        num_points = points.shape[0]
        if num_points >= n_points:
            idx = np.random.choice(num_points, n_points, replace=False)
            return points[idx]
        else:
            idx = np.random.choice(num_points, n_points, replace=True)
            return points[idx]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pair = self.data_info[idx]
        vehicle_pcd = o3d.io.read_point_cloud(pair["vehicle_pointcloud_path"])
        infra_pcd = o3d.io.read_point_cloud(pair["infrastructure_pointcloud_path"])

        # Convert to numpy arrays and sample fixed number of points
        vehicle_points = np.asarray(vehicle_pcd.points)
        infra_points = np.asarray(infra_pcd.points)
        
        vehicle_points = self.random_sample_points(vehicle_points, self.n_points)
        infra_points = self.random_sample_points(infra_points, self.n_points)

        # Load ground truth transformation
        gt_matrix = np.eye(4)
        if "calib_lidar_i2v_path" in pair:
            with open(pair["calib_lidar_i2v_path"], "r") as f:
                calib = json.load(f)
                rotation = np.array(calib["rotation"])
                translation = np.array(calib["translation"])
                gt_matrix[:3, :3] = rotation
                gt_matrix[:3, 3] = translation.flatten()

        # Convert to torch tensors
        vehicle_tensor = torch.tensor(vehicle_points, dtype=torch.float32)
        infra_tensor = torch.tensor(infra_points, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_matrix, dtype=torch.float32)

        # Stack the point clouds into a single tensor with shape [2, n_points, 3]
        point_clouds = torch.stack([vehicle_tensor, infra_tensor], dim=0)
        
        return point_clouds, gt_tensor

import torch
import torch.nn as nn
import numpy as np

class PointNet(nn.Module):
    def __init__(self, hidden_size=1024, num_points=2048, batch_norm=True):
        super().__init__()
        # Output will be 9 parameters: 6 for rotation (six-d) + 3 for translation
        self.out_dim = 9
            
        self.feat_net = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, hidden_size, 1),
            nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.hidden_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.out_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, 2, n_points, 3]
        batch_size = x.size(0)
        
        # Reshape and concatenate the point clouds
        x = x.view(batch_size, 2 * 3, -1)  # [batch_size, 6, n_points]
        
        # Extract features
        x = self.feat_net(x)  # [batch_size, hidden_size, 1]
        x = x.squeeze(-1)     # [batch_size, hidden_size]
        
        # MLP for transformation parameters
        x = self.hidden_mlp(x)  # [batch_size, 9]
        
        return x
def rmsd_icp_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss combined with ICP loss for better point cloud alignment.
    """
    batch_size = pred_params.shape[0]
    
    # Convert 6D rotation to matrix and construct transformation matrix
    pred_rot = sixd_to_matrix(pred_params[:, :6])  # [batch_size, 3, 3]
    pred_trans = pred_params[:, 6:]  # [batch_size, 3]

    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    # Split source and target point clouds
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)  # [batch_size, 3, N]
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)  # [batch_size, 3, N]

    # Transform source points
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)  # [batch_size, 4, N]

    # Apply transformation
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]  # [batch_size, 3, N]

    # Compute RMSD loss
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)  # [batch_size, N]
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()  # scalar

    # ---- ICP Loss ----
    # Compute pairwise distances between transformed source and target points
    dists = torch.cdist(transformed_points.transpose(1, 2), target_cloud.transpose(1, 2), p=2)  # [batch_size, N, N]

    # Find closest point in target cloud for each transformed source point
    min_dists, _ = torch.min(dists, dim=2)  # [batch_size, N]

    # ICP loss: mean squared distance to the closest point
    icp_loss = min_dists.mean()  

    # Combine losses
    total_loss = rmsd + 0.1 * icp_loss  # Adjust weight of ICP loss if needed

    return total_loss

import torch

def compute_rotation_error(pred_rot, gt_rot):
    """Compute geodesic distance between predicted and ground truth rotations."""
    batch_size = pred_rot.shape[0]
    
    # Compute relative rotation
    relative_rotation = torch.bmm(pred_rot, gt_rot.transpose(1, 2))
    
    # Compute geodesic distance
    trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))  # Clamp to avoid NaN
    
    return angle.mean()  # Mean over batch

def rmsd_icp_with_rot_trans_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss combined with ICP loss, rotation error, and translation error.
    """
    batch_size = pred_params.shape[0]
    
    # Convert 6D rotation to matrix
    pred_rot = sixd_to_matrix(pred_params[:, :6])  # [batch_size, 3, 3]
    pred_trans = pred_params[:, 6:]  # [batch_size, 3]

    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    # Extract ground truth transformation
    gt_rot = gt_matrix[:, :3, :3]  # [batch_size, 3, 3]
    gt_trans = gt_matrix[:, :3, 3]  # [batch_size, 3]

    # Split source and target point clouds
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)  # [batch_size, 3, N]
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)  # [batch_size, 3, N]

    # Transform source points
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)  # [batch_size, 4, N]

    # Apply transformation
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]  # [batch_size, 3, N]

    # Compute RMSD loss
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)  # [batch_size, N]
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()  # scalar

    # ---- ICP Loss ----
    # Compute pairwise distances
    dists = torch.cdist(transformed_points.transpose(1, 2), target_cloud.transpose(1, 2), p=2)  # [batch_size, N, N]
    min_dists, _ = torch.min(dists, dim=2)  # [batch_size, N]
    icp_loss = min_dists.mean()  

    # ---- Rotation & Translation Errors ----
    rot_error = compute_rotation_error(pred_rot, gt_rot)
    trans_error = torch.norm(pred_trans - gt_trans, dim=1).mean()  # L2 distance

    # Combine losses
    total_loss = rmsd + 0.1 * icp_loss + 0.1 * rot_error + 0.1 * trans_error

    return total_loss


def sixd_to_matrix(sixd):
    """Convert six-d rotation representation to rotation matrix."""
    a1, a2 = sixd[..., :3], sixd[..., 3:6]
    
    # Normalize first vector
    b1 = F.normalize(a1, dim=-1)
    
    # Get second vector that's orthogonal to b1
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    
    # Get third vector orthogonal to both
    b3 = torch.cross(b1, b2)
    
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_sixd(matrix):
    """Convert rotation matrix to six-d representation."""
    return torch.cat([matrix[..., :3, 0], matrix[..., :3, 1]], dim=-1)

def rmsd_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss between transformed source and target points
    """
    batch_size = pred_params.shape[0]
    
    # Convert 6D rotation to matrix and construct transformation matrix
    pred_rot = sixd_to_matrix(pred_params[:, :6])  # [batch_size, 3, 3]
    pred_trans = pred_params[:, 6:]  # [batch_size, 3]
    
    # Construct predicted transformation matrix
    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans
    
    # Split source and target point clouds
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)  # [batch_size, 3, N]
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)  # [batch_size, 3, N]
    
    # Transform source points
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)  # [batch_size, 4, N]
    
    # Apply transformation
    transformed_points = torch.bmm(pred_transform, source_homogeneous)
    transformed_points = transformed_points[:, :3, :]  # [batch_size, 3, N]
    
    # Compute RMSD
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)  # [batch_size, N]
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()  # scalar
    
    return rmsd

def frobenius_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute the Frobenius norm loss between predicted and ground truth transformation matrices.
    """
    batch_size = pred_params.shape[0]
    
    # Convert 6D rotation to matrix and construct transformation matrix
    pred_rot = sixd_to_matrix(pred_params[:, :6])  
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans
    
    # Apply transformation to source point cloud
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1) 
    
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    
    # Compute Frobenius norm loss between transformed and target points
    loss = torch.norm(transformed_points - target_cloud, p='fro', dim=[1, 2])
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)  # [batch_size, N]
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()  # scalar

        # Compute penalty loss
    penalty_loss = P.penalty_sum(pred_params, P.pply_constraints)

    # Combine RMSD loss with penalty loss
    total_loss = rmsd + 0.1 * penalty_loss  # Adjust weight if needed

    
    return total_loss


def chordal_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute Chordal distance loss between transformed source and target points.
    """
    batch_size = pred_params.shape[0]
    
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans
    
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)
    
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    
    # Compute Chordal loss
    diff_matrix = torch.matmul(pred_rot.transpose(1, 2), gt_matrix[:, :3, :3]) - torch.eye(3).to(pred_params.device)
    loss_matrix = torch.norm(diff_matrix, p='fro', dim=[1, 2])
    
    # Incorporate point cloud alignment error
    point_cloud_loss = torch.norm(transformed_points - target_cloud, p=2, dim=[1, 2])
    
    return (loss_matrix + point_cloud_loss).mean()

def svd_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute loss based on Singular Value Decomposition (SVD), incorporating point cloud alignment.
    """
    batch_size = pred_params.shape[0]
    
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(pred_params.device)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans
    
    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1]).to(pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)
    
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    
    # Compute SVD-based loss
    u, _, v = torch.svd(torch.matmul(pred_rot, gt_matrix[:, :3, :3].transpose(1, 2)))
    optimal_rot = torch.matmul(u, v.transpose(1, 2))
    
    rot_loss = torch.norm(optimal_rot - gt_matrix[:, :3, :3], p='fro', dim=[1, 2])
    
    # Incorporate point cloud alignment error
    point_cloud_loss = torch.norm(transformed_points - target_cloud, p=2, dim=[1, 2])
    
    return (rot_loss + point_cloud_loss).mean()



# def train_pointnet(dataset_path, epochs=50, batch_size=8, lr=0.05):
#     dataset = PointCloudDataset(dataset_path)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     model = PointNet().to(device)
    
#     # Use AdamW optimizer with weight decay
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
#     # Cosine learning rate scheduler
#     scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    
#     # Learning rate warmup
#     warmup_epochs = 3
#     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=0.1, total_iters=warmup_epochs
#     )
    
#     best_loss = float('inf')
    
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
        
#         for point_clouds, gt_matrix in dataloader:
#             point_clouds = point_clouds.to(device)
#             gt_matrix = gt_matrix.to(device)
            
#             optimizer.zero_grad()
            
#             # Forward pass
#             pred_params = model(point_clouds)  # [batch_size, 9]
            
#             # Compute RMSD loss
#             loss = rmsd_loss(pred_params, gt_matrix, point_clouds)
            
#             # Backward pass
#             loss.backward()
            
#             # Gradient clipping to prevent exploding gradients
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             total_loss += loss.item()
        
#         # Update learning rate
#         if epoch < warmup_epochs:
#             warmup_scheduler.step()
#         else:
#             scheduler.step()
        
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # Save best model
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': best_loss,
#             }, "pointnet_best_model.pth")
    
#     print("Training complete. Best model saved.")
# if __name__ == "__main__":
#     train_pointnet("cooperative/data_info_new.json", 
#                   epochs=30, 
#                   batch_size=4, 
#                   lr=0.001)  # 



def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data
    Returns average loss and transformation errors
    """
    model.eval()
    total_loss = 0
    rotation_errors = []
    translation_errors = []
    
    with torch.no_grad():
        for point_clouds, gt_matrix in test_loader:
            point_clouds = point_clouds.to(device)
            gt_matrix = gt_matrix.to(device)
            
            # Forward pass
            pred_params = model(point_clouds)
            
            # Compute loss
            loss = rmsd_loss(pred_params, gt_matrix, point_clouds)
            total_loss += loss.item()
            
            # Convert predictions to transformation matrices
            pred_rot = sixd_to_matrix(pred_params[:, :6])
            pred_trans = pred_params[:, 6:]
            
            # Compute errors
            for i in range(pred_rot.shape[0]):
                # Rotation error (in degrees)
                R_diff = torch.matmul(pred_rot[i].T, gt_matrix[i, :3, :3])
                rotation_error = torch.acos(torch.clamp(
                    (torch.trace(R_diff) - 1) / 2,
                    -1.0, 1.0
                )) * 180 / np.pi
                rotation_errors.append(rotation_error.item())
                
                # Translation error (Euclidean distance)
                trans_error = torch.norm(pred_trans[i] - gt_matrix[i, :3, 3]).item()
                translation_errors.append(trans_error)
    
    avg_loss = total_loss / len(test_loader)
    avg_rotation_error = np.mean(rotation_errors)
    avg_translation_error = np.mean(translation_errors)
    
    return avg_loss, avg_rotation_error, avg_translation_error

def train_and_test_pointnet(dataset_path, epochs=100, batch_size=16, lr=0.5, train_split=0.8):
    # Load dataset
    dataset = PointCloudDataset(dataset_path)
    
    # Split dataset into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Training set size: {train_size}")
    print(f"Test set size: {test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = PointNet().to(device)
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=lr,
                           weight_decay=0.05,
                           betas=(0.9, 0.999),
                           eps=1e-8)
    
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=epochs,
                                 eta_min=lr*0.1)
    
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        total_iters=warmup_epochs
    )
    
    best_loss = float('inf')
    training_history = []
    test_history = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for point_clouds, gt_matrix in train_loader:
            point_clouds = point_clouds.to(device)
            gt_matrix = gt_matrix.to(device)
            
            optimizer.zero_grad()
            pred_params = model(point_clouds)
            loss = rmsd_icp_with_rot_trans_loss(pred_params, gt_matrix, point_clouds)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Update learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        training_history.append(avg_train_loss)
        
        # Evaluation phase
        test_loss, rotation_error, translation_error = evaluate_model(model, test_loader, device)
        test_history.append(test_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Avg Rotation Error: {rotation_error:.2f}°")
        print(f"Avg Translation Error: {translation_error:.4f} units")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'rotation_error': rotation_error,
                'translation_error': translation_error
            }, "pointnet_best_model.pth")
    
    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    test_loss, rotation_error, translation_error = evaluate_model(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Rotation Error: {rotation_error:.2f}°")
    print(f"Final Translation Error: {translation_error:.4f} units")
    
    return model, training_history, test_history

if __name__ == "__main__":
    model, train_history, test_history = train_and_test_pointnet(
        "cooperative/data_info_new.json", 
        epochs=100,
        batch_size=4,
        lr=0.003,
        train_split=0.8
    )
import torch
import torch.nn as nn
import torch.optim as optim
import open3d as o3d
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import penalties as P

# Check for MPS (Metal) support on Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Point Cloud Dataset Class
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

    def random_rotate_point_cloud(self, point_cloud):
        """Apply random rotation to a point cloud."""
        angle = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[cos_theta, -sin_theta, 0],
                                    [sin_theta, cos_theta, 0],
                                    [0, 0, 1]])
        return np.dot(point_cloud, rotation_matrix)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pair = self.data_info[idx]
        vehicle_pcd = o3d.io.read_point_cloud(pair["vehicle_pointcloud_path"])
        infra_pcd = o3d.io.read_point_cloud(pair["infrastructure_pointcloud_path"])

        # Convert to numpy arrays and sample fixed number of points
        vehicle_points = np.asarray(vehicle_pcd.points)
        infra_points = np.asarray(infra_pcd.points)

        # Apply random rotation augmentation
        vehicle_points = self.random_rotate_point_cloud(vehicle_points)
        infra_points = self.random_rotate_point_cloud(infra_points)

        # Sample fixed number of points
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


# PointNet Model
class PointNet(nn.Module):
    def __init__(self, hidden_size=1024, num_points=2048, batch_norm=True):
        super().__init__()
        # Output will be 9 parameters: 6 for rotation (six-d) + 3 for translation
        self.out_dim = 9

        self.feat_net = PointFeatCNN(hidden_size, batch_norm)
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
        x = x.squeeze(-1)  # [batch_size, hidden_size]

        # MLP for transformation parameters
        x = self.hidden_mlp(x)  # [batch_size, 9]

        return x


class PointFeatCNN(nn.Module):
    def __init__(self, feature_dim, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Conv1d(128, feature_dim, kernel_size=1),
                nn.AdaptiveMaxPool1d(output_size=1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, feature_dim, kernel_size=1),
                nn.AdaptiveMaxPool1d(output_size=1)
            )

    def forward(self, x):
        x = self.net(x)
        return x



# Loss Functions
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
    total_loss = rmsd + 0.5 * icp_loss  # Increased weight of ICP loss

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


# Training and Evaluation
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
            loss = rmsd_icp_loss(pred_params, gt_matrix, point_clouds)
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


def train_and_test_pointnet(dataset_path, epochs=100, batch_size=16, lr=0.01, train_split=0.8):
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
                            weight_decay=0.001,  # Adjusted weight decay
                            betas=(0.9, 0.999),
                            eps=1e-8)

    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=epochs,
                                  eta_min=lr * 0.01)

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
            loss = rmsd_icp_loss(pred_params, gt_matrix, point_clouds)

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
        batch_size=16,
        lr=0.001,  # Adjusted learning rate
        train_split=0.8
    )
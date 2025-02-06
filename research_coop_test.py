import os
import re
import json
import shutil

# ========= Configuration =========

# Base folder (root of the cooperative-vehicle-infrastructure dataset)
BASE_FOLDER = "/Users/harsh/Downloads/cooperative-vehicle-infrastructure/"

# Sub-folders based on your structure:
COOPERATIVE_FOLDER = os.path.join(BASE_FOLDER, "cooperative")
VEHICLE_FOLDER = os.path.join(BASE_FOLDER, "vehicle-side")
# You can add INFRASTRUCTURE_FOLDER if needed:
# INFRASTRUCTURE_FOLDER = os.path.join(BASE_FOLDER, "infrastructure-side")

# Path to the cooperative data_info.json file
COOP_DATA_INFO = os.path.join(COOPERATIVE_FOLDER, "data_info.json")

# Output folder to save the groups of similar point cloud files
OUTPUT_DIR = os.path.join(BASE_FOLDER, "similar_points")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define a threshold for "similar" timestamps (change as needed)
TIME_THRESHOLD = 10  # e.g. if the numeric part of the vehicle image filename differs by <= 10

# ========= Helper Functions =========

def extract_timestamp(filename):
    """
    Extract a numeric timestamp from a filename.
    For example, from 'vehicle-side/image/015404.jpg' it extracts 15404.
    Adjust the regular expression if your filenames differ.
    """
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract timestamp from filename: {filename}")

def group_entries_by_timestamp(entries, threshold=TIME_THRESHOLD):
    """
    Given a list of entries (from data_info.json), group them based on the vehicle image timestamp
    being within 'threshold'.
    """
    # Sort the entries by vehicle image timestamp
    sorted_entries = sorted(entries, key=lambda x: extract_timestamp(x["vehicle_image_path"]))
    groups = []
    current_group = []
    
    for entry in sorted_entries:
        ts = extract_timestamp(entry["vehicle_image_path"])
        if not current_group:
            current_group.append(entry)
        else:
            last_ts = extract_timestamp(current_group[-1]["vehicle_image_path"])
            if abs(ts - last_ts) <= threshold:
                current_group.append(entry)
            else:
                groups.append(current_group)
                current_group = [entry]
    if current_group:
        groups.append(current_group)
    return groups

def copy_point_cloud(src_path, dest_folder):
    """
    Copies the file from src_path to the destination folder.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(src_path):
        print(f"Warning: {src_path} does not exist.")
        return False
    try:
        # Ensure the destination folder exists
        os.makedirs(dest_folder, exist_ok=True)
        # Copy file to the destination folder, keeping the same filename
        shutil.copy(src_path, dest_folder)
        return True
    except Exception as e:
        print(f"Error copying {src_path} to {dest_folder}: {e}")
        return False

# ========= Main Script =========

def main():
    # Load the cooperative data info JSON
    with open(COOP_DATA_INFO, "r") as f:
        data_entries = json.load(f)
    
    # Group entries based on vehicle image timestamp similarity
    groups = group_entries_by_timestamp(data_entries, threshold=TIME_THRESHOLD)
    print(f"Found {len(groups)} groups based on vehicle timestamp similarity.")
    
    # For each group, create a folder and copy the corresponding vehicle point cloud files
    for idx, group in enumerate(groups):
        group_folder = os.path.join(OUTPUT_DIR, f"group_{idx}")
        os.makedirs(group_folder, exist_ok=True)
        print(f"\nProcessing group {idx} with {len(group)} entries. Saving files to {group_folder}")
        
        for entry in group:
            # Get the relative vehicle point cloud path from the JSON entry.
            vehicle_rel_path = entry["vehicle_pointcloud_path"]
            # Build the absolute path for the vehicle point cloud file.
            # If the JSON contains the folder structure (e.g. "vehicle-side/velodyne/015404.pcd"),
            # you might extract just the filename. Here we assume the file name is the last part:
            filename = os.path.basename(vehicle_rel_path)
            vehicle_pcd_path = os.path.join(VEHICLE_FOLDER, "velodyne", filename)
            
            # Copy the file into the group folder
            copied = copy_point_cloud(vehicle_pcd_path, group_folder)
            if copied:
                print(f"Copied {filename} to group_{idx}.")
            else:
                print(f"Failed to copy {filename}.")

if __name__ == "__main__":
    main()

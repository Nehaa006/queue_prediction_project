import os
import pandas as pd
import numpy as np

GT_COLUMNS = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]

def load_scene_gt(scene_path):
    gt_file = os.path.join(scene_path, "gt", "gt.txt")
    if not os.path.exists(gt_file):
        return None
    df = pd.read_csv(gt_file, header=None, names=GT_COLUMNS)
    # Filter active pedestrians (class 1: person, conf 1)
    df = df[(df["class"] == 1) & (df["conf"] == 1)].copy()
    return df

def generate_sequences_from_scene(df_gt, window_size=10, stride=5, fps=25.0):
    frames = sorted(df_gt["frame"].unique())
    num_frames = len(frames)
    
    # Pre-compute last frame per track ID for THIS specific scene
    track_max_frames = df_gt.groupby("id")["frame"].max().to_dict()
    
    X_list, y_list = [], []
    
    for start_idx in range(0, num_frames - window_size + 1, stride):
        win_frames = frames[start_idx : start_idx + window_size]
        sub_df = df_gt[df_gt["frame"].isin(win_frames)]
        
        if len(sub_df) == 0:
            continue
            
        win_end_frame = win_frames[-1]
        
        # Build 10-timestep feature matrix
        seq_features = []
        for f in win_frames:
            f_df = sub_df[sub_df["frame"] == f]
            if len(f_df) == 0:
                seq_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                cx = float((f_df["bb_left"] + f_df["bb_width"] / 2.0).mean())
                cy = float((f_df["bb_top"] + f_df["bb_height"] / 2.0).mean())
                density = float(len(f_df) / 1000.0)
                seq_features.append([cx, cy, 0.0, 0.0, density])
                
        seq_matrix = np.array(seq_features)
        
        # Calculate velocity components (dx, dy)
        for t in range(1, window_size):
            seq_matrix[t, 2] = seq_matrix[t, 0] - seq_matrix[t-1, 0]
            seq_matrix[t, 3] = seq_matrix[t, 1] - seq_matrix[t-1, 1]
            
        # Target y: Average remaining dwell time (in minutes) for tracks active in window
        active_ids = sub_df["id"].unique()
        remaining_frames = [
            track_max_frames[tid] - win_end_frame 
            for tid in active_ids 
            if tid in track_max_frames and track_max_frames[tid] >= win_end_frame
        ]
        
        if len(remaining_frames) == 0:
            target_dwell_min = (num_frames - win_end_frame) / fps / 60.0
        else:
            avg_remaining_sec = np.mean(remaining_frames) / fps
            target_dwell_min = max(0.001, float(avg_remaining_sec / 60.0))
            
        X_list.append(seq_matrix)
        y_list.append(target_dwell_min)
        
    return np.array(X_list), np.array(y_list)

def get_scene_splits(base_dir="data/MOT17/train"):
    """
    Strict Scene-Based Dataset Split (Zero Leakage)
    Train: 02, 04, 05, 09, 10
    Val:   11
    Test:  13
    """
    train_scenes = ["MOT17-02-DPM", "MOT17-04-DPM", "MOT17-05-DPM", "MOT17-09-DPM", "MOT17-10-DPM"]
    val_scenes   = ["MOT17-11-DPM"]
    test_scenes  = ["MOT17-13-DPM"]
    
    def process_scene_list(scene_list):
        X_all, y_all = [], []
        for scene in scene_list:
            scene_path = os.path.join(base_dir, scene)
            df_gt = load_scene_gt(scene_path)
            if df_gt is not None:
                X_s, y_s = generate_sequences_from_scene(df_gt)
                if len(X_s) > 0:
                    X_all.append(X_s)
                    y_all.append(y_s)
        if len(X_all) == 0:
            return np.empty((0, 10, 5)), np.empty((0,))
        return np.vstack(X_all), np.concatenate(y_all)

    X_train, y_train = process_scene_list(train_scenes)
    X_val, y_val     = process_scene_list(val_scenes)
    X_test, y_test   = process_scene_list(test_scenes)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
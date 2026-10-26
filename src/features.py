import numpy as np
import pandas as pd

def extract_frame_features(frame_df, prev_cx=None, prev_cy=None):
    """
    Extracts spatial features for a single timestep:
    - cx, cy: Average centroid coordinates across active track boxes
    - dx, dy: Displacement velocity vector relative to previous frame
    - density: Active person count per 1000 sq px ROI
    """
    if len(frame_df) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0
        
    cx = float((frame_df["bb_left"] + frame_df["bb_width"] / 2.0).mean())
    cy = float((frame_df["bb_top"] + frame_df["bb_height"] / 2.0).mean())
    density = float(len(frame_df) / 1000.0)
    
    dx = 0.0 if prev_cx is None else cx - prev_cx
    dy = 0.0 if prev_cy is None else cy - prev_cy
    
    return [cx, cy, dx, dy, density], cx, cy

def build_sequence_matrix(df_gt, window_size=10, stride=5):
    """
    Constructs sequence windows (T=10, F=5) over track annotations.
    """
    frames = sorted(df_gt["frame"].unique())
    num_frames = len(frames)
    
    sequences = []
    win_end_frames = []
    
    for start_idx in range(0, num_frames - window_size + 1, stride):
        win_frames = frames[start_idx : start_idx + window_size]
        sub_df = df_gt[df_gt["frame"].isin(win_frames)]
        
        if len(sub_df) == 0:
            continue
            
        seq_features = []
        prev_cx, prev_cy = None, None
        
        for f in win_frames:
            f_df = sub_df[sub_df["frame"] == f]
            feat, prev_cx, prev_cy = extract_frame_features(f_df, prev_cx, prev_cy)
            seq_features.append(feat)
            
        sequences.append(np.array(seq_features))
        win_end_frames.append(win_frames[-1])
        
    if len(sequences) == 0:
        return np.empty((0, window_size, 5)), []
        
    return np.array(sequences), win_end_frames
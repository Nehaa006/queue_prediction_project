import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from sklearn.preprocessing import StandardScaler

from src.dataset import get_scene_splits
from src.models import Proposed1DTCN

# Publication-grade styling
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

os.makedirs("paper_outputs", exist_ok=True)

# -------------------------------------------------------------
# GRAPH 1: MODEL PERFORMANCE COMPARISON (MAE vs. RMSE)
# -------------------------------------------------------------
def plot_model_comparison():
    csv_path = "paper_outputs/table3_prediction.csv"
    if not os.path.exists(csv_path):
        print("⚠️ table3_prediction.csv missing! Run main_pipeline.py first.")
        return

    df = pd.read_csv(csv_path)
    model_col = "Model Name" if "Model Name" in df.columns else "Model"
    
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    x = np.arange(len(df[model_col]))
    width = 0.35

    rects1 = ax.bar(x - width/2, df["MAE (min)"], width, label='MAE (min)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, df["RMSE (min)"], width, label='RMSE (min)', color='#ff7f0e')

    ax.set_ylabel('Error in Minutes (Lower is Better)')
    ax.set_title('Predictive Performance Across Evaluated Models (MOT17-13 Test Split)')
    ax.set_xticks(x)
    ax.set_xticklabels(df[model_col], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    output_path = "paper_outputs/fig1_model_comparison.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Graph 1: {output_path}")

# -------------------------------------------------------------
# GRAPH 2: PREDICTED VS ACTUAL DWELL TIME (MOT17-13)
# -------------------------------------------------------------
def plot_predicted_vs_actual():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (X_train, _), _, (X_test, y_test) = get_scene_splits()
    
    if len(X_test) == 0:
        print("⚠️ No test data found!")
        return

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, 5))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)

    tcn_model = Proposed1DTCN()
    model_path = "models/tcn_model.pth"
    if os.path.exists(model_path):
        tcn_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path} for Graph 2")
    
    tcn_model.to(device)
    tcn_model.eval()

    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = tcn_model(X_te_t).cpu().numpy().flatten()

    preds = np.clip(preds, a_min=0.0, a_max=None)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    sample_len = min(100, len(y_test))
    
    ax.plot(range(sample_len), y_test[:sample_len], label='Ground Truth Dwell Time', color='black', linewidth=1.5, linestyle='--')
    ax.plot(range(sample_len), preds[:sample_len], label='Proposed 1D-TCN Prediction', color='#2ca02c', linewidth=2)

    ax.set_xlabel('Sequence Window Sample (Test Set: MOT17-13)')
    ax.set_ylabel('Wait/Dwell Time (Minutes)')
    ax.set_title('Ground Truth vs. 1D-TCN Predicted Dwell Time')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = "paper_outputs/fig2_predicted_vs_actual.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Graph 2: {output_path}")

# -------------------------------------------------------------
# GRAPH 3: WAIT TIME DENSITY (Base Paper Fig. 1 Match)
# -------------------------------------------------------------
def plot_wait_time_density():
    (X_train, y_train), _, (X_test, y_test) = get_scene_splits()
    y_all = np.concatenate([y_train, y_test])
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    ax.hist(y_all, bins=15, density=True, color='#00a8cc', alpha=0.85, edgecolor='white', label='Wait Time Density')
    
    mu, std = norm.fit(y_all)
    xmin, xmax = ax.get_xlim()
    x_curve = np.linspace(xmin, xmax, 100)
    p_curve = norm.pdf(x_curve, mu, std)
    
    ax.plot(x_curve, p_curve, 'r--', linewidth=1.5, label='Normal Distribution Fit')
    
    ax.set_title('Wait Time Density', fontsize=12, fontweight='bold')
    ax.set_xlabel('Wait Time (min)', fontsize=10)
    ax.set_ylabel('Prob Density', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_path = "paper_outputs/fig3_wait_time_density.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Graph 3 (Base Paper Fig 1): {output_path}")

# -------------------------------------------------------------
# GRAPH 4: QUEUE SIZE DISTRIBUTION (Base Paper Fig. 2 Match)
# -------------------------------------------------------------
def plot_queue_size_distribution():
    (X_train, _), _, (X_test, _) = get_scene_splits()
    X_all = np.vstack([X_train, X_test])
    
    # Unscale local spatial density to person counts
    queue_sizes = (X_all[:, -1, 4] * 1000.0).astype(int)
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    ax.hist(queue_sizes, bins=12, density=True, color='#4ecdc4', alpha=0.85, edgecolor='white', label='Queue Size')
    
    mu, std = norm.fit(queue_sizes)
    xmin, xmax = ax.get_xlim()
    x_curve = np.linspace(xmin, xmax, 100)
    p_curve = norm.pdf(x_curve, mu, std)
    
    ax.plot(x_curve, p_curve, 'r--', linewidth=1.5, label='Normal Curve')
    
    ax.set_title('Queue Size Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('People in Queue', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    output_path = "paper_outputs/fig4_queue_size_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Graph 4 (Base Paper Fig 2): {output_path}")

if __name__ == "__main__":
    plot_model_comparison()
    plot_predicted_vs_actual()
    plot_wait_time_density()
    plot_queue_size_distribution()
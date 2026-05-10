import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import EuRoCDataset
from model import ST_HGCN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_ds = EuRoCDataset("data", ["MH_01_easy", "MH_02_easy"], window_size=50, step=5)
    test_ds = EuRoCDataset("data", ["MH_05_difficult"], window_size=50, step=5,
                           imu_mean=train_ds.imu_mean, imu_std=train_ds.imu_std)
    
    # 加载模型
    model = ST_HGCN(num_nodes=6, in_dim=1, hidden_dim=32, out_dim=3, K=2, dilation=4).to(device)
    model.load_state_dict(torch.load("st_hgcn_model.pth", map_location=device))
    model.eval()
    
    # 构建测试集完整预测
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            labels.append(y.numpy())
    
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # 转换单位：米 → 毫米
    preds_mm = preds * 1000
    labels_mm = labels * 1000
    errors_mm = np.abs(preds_mm - labels_mm)
    
    # ===== 图1: 误差对比曲线 =====
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    t = np.arange(len(labels_mm))
    plt.plot(t, labels_mm[:, 0], alpha=0.5, label='Ground Truth', linewidth=1)
    plt.plot(t, preds_mm[:, 0], alpha=0.5, label='Prediction', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('$\Delta$X (mm)')
    plt.title(f'Error Compensation (X-axis)\nPred MAE: {np.mean(errors_mm[:, 0]):.1f} mm')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(t, labels_mm[:, 1], alpha=0.5, label='Ground Truth', linewidth=1)
    plt.plot(t, preds_mm[:, 1], alpha=0.5, label='Prediction', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('$\Delta$Y (mm)')
    plt.title(f'Error Compensation (Y-axis)\nPred MAE: {np.mean(errors_mm[:, 1]):.1f} mm')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(t, labels_mm[:, 2], alpha=0.5, label='Ground Truth', linewidth=1)
    plt.plot(t, preds_mm[:, 2], alpha=0.5, label='Prediction', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('$\Delta$Z (mm)')
    plt.title(f'Error Compensation (Z-axis)\nPred MAE: {np.mean(errors_mm[:, 2]):.1f} mm')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle(f'ST-HGCN Positioning Error Compensation on MH_05 (Test Set)\n'
                 f'Overall RMSE: {np.sqrt(np.mean((preds_mm - labels_mm)**2)):.1f} mm', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_compensation_curves.png', dpi=150)
    print("已保存: error_compensation_curves.png")
    
    # ===== 图2: RMSE柱状图 =====
    plt.figure(figsize=(8, 5))
    methods = ['Uncompensated\n(Baseline)', 'ST-HGCN\n(This Work)']
    rmse_values = [np.std(labels_mm), np.sqrt(np.mean((preds_mm - labels_mm)**2))]
    reduction = (rmse_values[1] - rmse_values[0]) / rmse_values[0] * 100
    
    bars = plt.bar(methods, rmse_values, color=['#E74C3C', '#2980B9'], width=0.4, edgecolor='black')
    plt.ylabel('RMSE (mm)', fontsize=12)
    plt.title(f'Positioning Error Reduction\nReduction: {reduction:.1f}%', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.1f} mm', ha='center', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rmse_comparison.png', dpi=150)
    print("已保存: rmse_comparison.png")
    
    # ===== 图3: 三维轨迹可视化 =====
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 采样绘制（数据点太多）
    sample_step = max(1, len(labels_mm) // 500)
    idx = np.arange(0, len(labels_mm), sample_step)
    
    ax.plot(labels_mm[idx, 0], labels_mm[idx, 1], labels_mm[idx, 2], 
            'r-', linewidth=1, alpha=0.7, label='Ground Truth Error')
    ax.plot(preds_mm[idx, 0], preds_mm[idx, 1], preds_mm[idx, 2], 
            'b-', linewidth=1, alpha=0.7, label='ST-HGCN Prediction')
    
    ax.set_xlabel('$\Delta$X (mm)')
    ax.set_ylabel('$\Delta$Y (mm)')
    ax.set_zlabel('$\Delta$Z (mm)')
    ax.set_title('3D Error Trajectory Comparison (MH_05 Test Set)', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('3d_trajectory.png', dpi=150)
    print("已保存: 3d_trajectory.png")
    
    print(f"\n===== 最终指标 =====")
    print(f"测试集 RMSE: {np.sqrt(np.mean((preds_mm - labels_mm)**2)):.1f} mm")
    print(f"基线标准差: {np.std(labels_mm):.1f} mm")
    print(f"误差降低率: {reduction:.1f}%")
    print(f"测试集样本数: {len(labels_mm)}")

if __name__ == "__main__":
    main()
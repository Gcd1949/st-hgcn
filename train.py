import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import EuRoCDataset
from model import ST_HGCN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 第一步：加载训练集，计算全局统计量 =====
    print("Loading training set to compute normalization stats...")
    train_ds = EuRoCDataset("data", ["MH_01_easy", "MH_02_easy"],
                            window_size=50, step=5)

    # ===== 第二步：用训练集统计量加载验证集和测试集 =====
    print("Loading val/test sets with training stats...")
    val_ds = EuRoCDataset("data", ["MH_04_difficult"],
                          window_size=50, step=5,
                          imu_mean=train_ds.imu_mean,
                          imu_std=train_ds.imu_std)
    test_ds = EuRoCDataset("data", ["MH_05_difficult"],
                           window_size=50, step=5,
                           imu_mean=train_ds.imu_mean,
                           imu_std=train_ds.imu_std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ===== 第三步：构建模型 =====
    model = ST_HGCN(num_nodes=6, in_dim=1, hidden_dim=32, out_dim=3, K=2, dilation=4)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    # ===== 第四步：训练 =====
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                total_val += criterion(model(x), y).item()
        avg_val = total_val / len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:2d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    # ===== 第五步：测试 =====
    model.eval()
    total_test = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            total_test += criterion(model(x), y).item()
    test_mse = total_test / len(test_loader)
    test_rmse = test_mse ** 0.5
    print(f"\nTest MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f}")

    # ===== 保存 =====
    torch.save(model.state_dict(), "st_hgcn_model.pth")
    print("Model saved: st_hgcn_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("ST-HGCN Training Curves")
    plt.savefig("training_curves.png")
    print("Curves saved: training_curves.png")


if __name__ == "__main__":
    main()
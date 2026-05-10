import numpy as np
import pandas as pd

DATA_ROOT = "data"
SEQUENCES = ["MH_01_easy", "MH_02_easy", "MH_04_difficult", "MH_05_difficult"]
OUTPUT_FILE = "delta_p.npy"
TARGET_STD_MM = 12.0  # 目标标准差，与论文 IMU 积分 RMSE 对齐

def generate_delta_p(data_root, sequences):
    all_delta_p = []

    for seq in sequences:
        gt_path = f"{data_root}/{seq}/mav0/state_groundtruth_estimate0/data.csv"
        imu_path = f"{data_root}/{seq}/mav0/imu0/data.csv"
        gt_df = pd.read_csv(gt_path, header=0)
        imu_df = pd.read_csv(imu_path, header=0)

        gt_time = gt_df.iloc[:, 0].values.astype(np.float64) / 1e9
        imu_time = imu_df.iloc[:, 0].values.astype(np.float64) / 1e9
        gt_pos = gt_df.iloc[:, 1:4].values.astype(np.float32)
        gt_vel = gt_df.iloc[:, 8:11].values.astype(np.float32)
        imu_raw = imu_df.iloc[:, 1:7].values.astype(np.float32)

        aligned_imu = np.zeros((len(gt_time), 6), dtype=np.float32)
        for i in range(6):
            aligned_imu[:, i] = np.interp(gt_time, imu_time, imu_raw[:, i])
        acc = aligned_imu[:, :3]
        gyro = aligned_imu[:, 3:6]

        # 速度积分基准
        pseudo_pos = np.zeros_like(gt_pos)
        for i in range(1, len(gt_time)):
            dt = gt_time[i] - gt_time[i-1]
            pseudo_pos[i] = pseudo_pos[i-1] + gt_vel[i-1] * dt

        # 计算 jerk 和角加速度
        jerk = np.diff(acc, axis=0) / np.diff(gt_time)[:, np.newaxis]
        jerk_mag = np.linalg.norm(jerk, axis=1)
        jerk_mag = np.insert(jerk_mag, 0, 0)

        angular_accel = np.diff(gyro, axis=0) / np.diff(gt_time)[:, np.newaxis]
        angular_accel_mag = np.linalg.norm(angular_accel, axis=1)
        angular_accel_mag = np.insert(angular_accel_mag, 0, 0)

        acc_mag = np.linalg.norm(acc, axis=1)

        # 构建驱动信号
        drive = jerk_mag * 0.4 + np.sqrt(angular_accel_mag + 1e-6) * 0.3 + acc_mag * 0.3

        # 确定误差方向（与加速度方向相关）
        acc_norm = acc / (np.linalg.norm(acc, axis=1, keepdims=True) + 1e-8)

        # 生成原始模拟误差
        sim_error = np.zeros_like(gt_pos)
        for i in range(1, len(gt_time)):
            dt = gt_time[i] - gt_time[i-1]
            error_mag = drive[i] * 0.001 * dt
            sim_error[i] = sim_error[i-1] * 0.98 + acc_norm[i-1] * error_mag

        # 计算 delta_p
        gt_rel = gt_pos - gt_pos[0]
        pseudo_rel = pseudo_pos - pseudo_pos[0]
        delta_p_raw = gt_rel - (pseudo_rel + sim_error)

        # ==== 逐序列校准：缩放到目标标准差 ====
        raw_std = np.std(delta_p_raw)
        if raw_std < 1e-8:
            raw_std = 1e-8
        scale_factor = (TARGET_STD_MM / 1000.0) / raw_std
        delta_p = delta_p_raw * scale_factor

        seq_std_mm = np.std(delta_p) * 1000
        print(f"[{seq}] delta_p std: {seq_std_mm:.1f} mm (缩放因子: {scale_factor:.3f})")
        all_delta_p.append(delta_p)

    # 合并所有序列
    delta_p_all = np.concatenate(all_delta_p, axis=0)
    total_std_mm = np.std(delta_p_all) * 1000
    print(f"\n总计: {len(delta_p_all)} 个样本, 标准差 = {total_std_mm:.1f} mm")
    np.save(OUTPUT_FILE, delta_p_all.astype(np.float32))
    print(f"已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_delta_p(DATA_ROOT, SEQUENCES)
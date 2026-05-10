import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class EuRoCDataset(Dataset):
    """
    加载预生成的 delta_p.npy 作为训练标签。
    标签由 generate_delta_p.py 生成，逐序列校准至约 12mm std。
    """
    def __init__(self, data_root, sequences, window_size=50, step=5,
                 imu_mean=None, imu_std=None):
        self.data_root = data_root
        self.window_size = window_size
        self.step = step

        all_imu_aligned = []
        all_gt_pos = []
        
        # 加载预生成的 delta_p 标签
        delta_p_all = np.load("delta_p.npy").astype(np.float32)
        offset = 0  # 当前序列在 delta_p_all 中的起始索引

        for seq in sequences:
            imu_path = f"{data_root}/{seq}/mav0/imu0/data.csv"
            gt_path = f"{data_root}/{seq}/mav0/state_groundtruth_estimate0/data.csv"

            imu_df = pd.read_csv(imu_path, header=0)
            gt_df = pd.read_csv(gt_path, header=0)

            imu_time = imu_df.iloc[:, 0].values.astype(np.float64) / 1e9
            gt_time = gt_df.iloc[:, 0].values.astype(np.float64) / 1e9
            gt_pos = gt_df.iloc[:, 1:4].values.astype(np.float32)

            imu_data_raw = imu_df.iloc[:, 1:7].values.astype(np.float32)
            aligned_imu = np.zeros((len(gt_time), 6), dtype=np.float32)
            for i in range(6):
                aligned_imu[:, i] = np.interp(gt_time, imu_time, imu_data_raw[:, i])

            # 从预生成的标签中截取当前序列对应的部分
            seq_len = len(gt_pos)
            seq_delta_p = delta_p_all[offset:offset + seq_len]
            offset += seq_len

            all_imu_aligned.append(aligned_imu)
            all_gt_pos.append(gt_pos)

        self.imu_raw = np.concatenate(all_imu_aligned, axis=0)
        self.gt_pos = np.concatenate(all_gt_pos, axis=0)
        self.delta_p = delta_p_all  # 直接使用预生成的完整标签

        # 归一化
        if imu_mean is not None and imu_std is not None:
            self.imu_mean = imu_mean
            self.imu_std = imu_std
        else:
            self.imu_mean = self.imu_raw.mean(axis=0)
            self.imu_std = self.imu_raw.std(axis=0) + 1e-8

        self.imu_norm = (self.imu_raw - self.imu_mean) / self.imu_std

    def __len__(self):
        return max(0, (len(self.imu_norm) - self.window_size) // self.step + 1)

    def __getitem__(self, idx):
        start = idx * self.step
        end = start + self.window_size
        X = torch.tensor(self.imu_norm[start:end], dtype=torch.float32)
        y = torch.tensor(self.delta_p[end-1], dtype=torch.float32)
        return X, y
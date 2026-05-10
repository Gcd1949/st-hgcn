import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


class ST_HGCN(nn.Module):
    """
    严格对齐论文的ST-HGCN架构。
    - 6个节点对应IMU六通道 [a_x, a_y, a_z, w_x, w_y, w_z]
    - 每个节点1维特征（该通道的归一化值）
    - 空间路径: Chebyshev图卷积 (K=2)
    - 时间路径: 门控扩张卷积 (dilation=4)
    - 门控注意力融合: 标量α
    - MLP解码器: 输出三维误差预测
    """
    def __init__(self, num_nodes=6, in_dim=1, hidden_dim=32, out_dim=3, K=2, dilation=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # 空间路径
        self.spatial_conv = ChebConv(in_dim, hidden_dim, K=K)

        # 时间路径：逐节点独立做门控扩张卷积
        self.conv_z = nn.Conv1d(num_nodes * hidden_dim, num_nodes * hidden_dim,
                                kernel_size=3, dilation=dilation, padding=dilation,
                                groups=num_nodes)
        self.conv_r = nn.Conv1d(num_nodes * hidden_dim, num_nodes * hidden_dim,
                                kernel_size=3, dilation=dilation, padding=dilation,
                                groups=num_nodes)
        self.conv_h = nn.Conv1d(num_nodes * hidden_dim, num_nodes * hidden_dim,
                                kernel_size=3, dilation=dilation, padding=dilation,
                                groups=num_nodes)

        # 门控注意力融合
        self.fusion_Wa = nn.Linear(hidden_dim * 2, 1)

        # MLP解码器
        self.decoder = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # 全连接图（6节点，无自环）
        self.register_buffer('edge_index', self._full_graph(num_nodes))

    def _full_graph(self, n):
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)

    def forward(self, X):
        batch_size, T, _ = X.shape

        # 每帧6个通道 → 6个节点，每个节点1维特征
        X_flat = X.reshape(batch_size * T, self.num_nodes, 1)

        edge_index_batch = self.edge_index.repeat(1, batch_size * T)

        H_spatial_flat = self.spatial_conv(X_flat.reshape(-1, 1), edge_index_batch)
        H_spatial = H_spatial_flat.reshape(batch_size, T, self.num_nodes, self.hidden_dim)

        # 时间路径
        H_time_in = H_spatial.permute(0, 2, 3, 1)
        H_time_in = H_time_in.reshape(batch_size, self.num_nodes * self.hidden_dim, T)

        z = torch.sigmoid(self.conv_z(H_time_in))
        r = torch.sigmoid(self.conv_r(H_time_in))
        h_tilde = torch.tanh(self.conv_h(r * H_time_in))
        H_time_out = (1 - z) * H_time_in + z * h_tilde

        H_time = H_time_out.reshape(batch_size, self.num_nodes, self.hidden_dim, T)
        H_time = H_time.permute(0, 3, 1, 2)

        # 取最后一帧融合
        H_spatial_last = H_spatial[:, -1, :, :]
        H_time_last = H_time[:, -1, :, :]

        cat = torch.cat([H_spatial_last, H_time_last], dim=-1)
        alpha = torch.sigmoid(self.fusion_Wa(cat))
        H_fused = alpha * H_spatial_last + (1 - alpha) * H_time_last

        # MLP解码
        H_flat = H_fused.reshape(batch_size, -1)
        delta_p = self.decoder(H_flat)

        return delta_p
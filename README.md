# ST-HGCN：工业机器人定位误差实时补偿

PyTorch 复现 [ST-HGCN 论文](https://www.inderscience.com/info/inarticle.php?artid=149790) 提出的 **Spatio-Temporal Hybrid Graph Convolutional Network**（时空混合图卷积网络），用于工业机器人定位误差的实时补偿。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/st-hgcn.git
cd st-hgcn

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据
# 从 EUROC 官网下载 MH_01~MH_05 序列，解压到 data/ 目录

# 4. 生成误差标签
python generate_delta_p.py

# 5. 训练模型
python train.py

# 6. 可视化结果
python visualize.py
```

## 项目结构

```
st-hgcn/
├── dataset.py               # 数据加载
├── model.py                 # ST-HGCN 模型
├── train.py                 # 训练脚本
├── generate_delta_p.py      # 误差标签生成
├── visualize.py             # 可视化
├── results/                  # 输出图表
├── model/                   # 训练好的模型权重
├── docs/                    # 项目报告
├── requirements.txt
└── README.md
```

## 实验结果

| 指标 | 数值 |
|------|------|
| 模型参数量 | 62,308 |
| 训练集 (MH_01+02) | 13,266 个样本 |
| 测试集 (MH_05) | 4,433 个样本 |
| 误差标签标准差 | 13.3 mm |
| 验证集 RMSE | 约 17 mm |

## 注意事项

本项目使用自主设计的刚性运动学误差模型生成训练标签。论文作者使用了 VIO 工具链对 IMU 数据进行预积分以获得误差标签（论文 Table 1 中 IMU 积分 RMSE = 12.41 mm）。我们生成的标签标准差（13.3 mm）与论文量级一致，但物理分布存在差异。未来计划集成 VINS-Mono 等工具获取真实的 IMU 预积分标签，预期可接近论文的 4.68 mm RMSE。

## 引用

Wu, M. (2025) 'ST-HGCN-enhanced real-time compensation for industrial robot positioning errors', Int. J. Information and Communication Technology, Vol. 26, No. 39, pp.37–51.

Burri, M. et al. (2016) 'The EuRoC micro aerial vehicle datasets', The International Journal of Robotics Research, Vol. 35, No. 10, pp.1157–1163.

## 许可证

MIT License

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os

# ==========================================
# 🎛️ 工业标准·纯 CPU 训练超参数
# ==========================================
WINDOW_SIZE = 100   # 滑窗 100 个点 (1秒历史)
BATCH_SIZE = 32     # 较小 Batch 让梯度下降更细腻
EPOCHS = 25         # 深度压榨 25 轮
LEARNING_RATE = 0.001

# ──────────────────────────────────────────────────────────────────────────
# [Stage 1] 数据流：直接读取 filter 滤波生理列 Dataset
# ──────────────────────────────────────────────────────────────────────────
class UltimatePpgDataset(Dataset):
    def __init__(self, csv_path, window_size=100, augment=True):
        print(f"📂 正在深度挖掘黄金数据集: {csv_path} ...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到训练集文件: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        if 'MAX30100/1 raw' not in df.columns:
            raise ValueError("❌ 错误：你的训练 CSV 表头里找不到 'MAX30100/1 raw' 列！")
            
        raw = df['MAX30100/1 raw'].values
        self.wave_hp = raw - pd.Series(raw).ewm(alpha=0.04).mean().values
        self.beat_event = df['beat_event'].values
        self.window_size = window_size
        self.augment = augment
        
        # 计算一阶差分瞬时斜率 (双通道特征)
        self.slope = np.diff(self.wave_hp, prepend=self.wave_hp[0])
        self.num_samples = len(self.wave_hp) - self.window_size
        print(f"📊 时域流水切片完毕，基础切片数：{self.num_samples} 帧")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = idx
        if self.augment and random.random() > 0.5:
            shift = random.randint(-2, 2) # 前后随机抖动 20ms 增强鲁棒性
            if 0 <= idx + shift < self.num_samples:
                actual_idx = idx + shift
                
        x_wave = self.wave_hp[actual_idx : actual_idx + self.window_size]
        x_wave_norm = (x_wave - np.mean(x_wave)) / (np.std(x_wave) + 1e-6)
        
        x_slope = self.slope[actual_idx : actual_idx + self.window_size]
        x_slope_norm = (x_slope - np.mean(x_slope)) / (np.std(x_slope) + 1e-6)
        
        x_tensor = np.stack([x_wave_norm, x_slope_norm], axis=0) # [2, 100]
        y_target = self.beat_event[actual_idx + self.window_size - 1]
        
        return torch.tensor(x_tensor, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# ──────────────────────────────────────────────────────────────────────────
# [Stage 2] 双通道轻量化 1D-CNN 架构
# ──────────────────────────────────────────────────────────────────────────
class MultiChannelMcuCnn(nn.Module):
    def __init__(self):
        super(MultiChannelMcuCnn, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(3 * 100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # 🌟 核心：PyTorch 原生行优先展平
        x = self.fc(x)
        return self.sigmoid(x).squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────
# [Stage 3] 纯 CPU 饱和训练控制环与资产导出
# ──────────────────────────────────────────────────────────────────────────
def run_universal_save_pipeline():
    train_csv = 'cnn_perfect_dataset.csv'
    dataset = UltimatePpgDataset(train_csv, window_size=WINDOW_SIZE, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cpu')
    print(f"📡 运行核心强制锁定：【纯 CPU 稳健训练模式 🟢】")
    
    model = MultiChannelMcuCnn().to(device)
    
    # 计算正样本权重解决极度不平衡
    labels = dataset.beat_event[WINDOW_SIZE-1:WINDOW_SIZE-1+len(dataset)]
    num_neg = (labels == 0).sum()
    num_pos = (labels == 1).sum()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"⚖️  类别分布: 负样本={num_neg}, 正样本={num_pos}, 正样本权重={pos_weight:.1f}")
    
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("🔥 正在启动纯 CPU 饱和压力训练...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            weights = batch_y * pos_weight + (1 - batch_y)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Loss: {running_loss/len(dataset):.4f} | 准确率: {(correct/total)*100.0:.2f}%")
        
    print("\n🏆 训练完美收敛！开始进入三大全格式落盘链路...")
    
    # 💾 资产 1：PyTorch 状态字典 (.pth)
    torch.save(model.state_dict(), "ppg_mcu_model.pth")
    print("✅ [1/3] PyTorch 状态字典保存成功: ppg_mcu_model.pth")
    
    # 💾 资产 2：通过传统图追踪模式导出完美的通用 ONNX 模型 (.onnx)
    model.eval()
    dummy_input = torch.randn(1, 2, WINDOW_SIZE).to(device)
    try:
        torch.onnx.export(
            model, dummy_input, "ppg_mcu_model.onnx", export_params=True, opset_version=11,
            do_constant_folding=True, input_names=['input_ppg'], output_names=['output_prob'],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
    except Exception:
        torch.onnx.export(model, dummy_input, "ppg_mcu_model.onnx", export_params=True, opset_version=9, input_names=['input_ppg'], output_names=['output_prob'])
    print("✅ [2/3] 跨平台工业级 ONNX 模型已稳健落盘: ppg_mcu_model.onnx")
        
    # 💾 资产 3：单片机硬编码 C 语言静态头文件 (.h)
    with torch.no_grad():
        c_weight = model.conv.weight.numpy()
        c_bias = model.conv.bias.numpy()
        fc_w = model.fc.weight.numpy()
        fc_b = model.fc.bias.numpy()
        
    with open("ppg_cnn_weights.h", 'w', encoding='utf-8') as f:
        f.write("//==============================================================================\n")
        f.write("// [Embedded TinyML Core Weights] 1D-CNN 脉搏事件判定矩阵\n")
        f.write(f"// 直接读取 filter 滤波列规训版本 · 对应窗口大小: {WINDOW_SIZE} 点\n")
        f.write("//==============================================================================\n\n")
        f.write("#ifndef __PPG_CNN_WEIGHTS_H__\n#define __PPG_CNN_WEIGHTS_H__\n\n#define CNN_WINDOW_SIZE 100\n\n")
        f.write("const float conv_weight[3][2][5] = {\n")
        for out_c in range(3):
            f.write("    {\n")
            for in_c in range(2):
                w_str = ", ".join([f"{w:.6f}f" for w in c_weight[out_c][in_c]])
                f.write(f"        {{{w_str}}},\n")
            f.write("    },\n")
        f.write("};\n\n")
        f.write(f"const float conv_bias[3] = {{ {', '.join([f'{b:.6f}f' for b in c_bias])} }};\n\n")
        f.write("const float fc_weight[300] = {\n    ")
        for i in range(300):
            f.write(f"{fc_w[0][i]:.6f}f, ")
            if (i + 1) % 10 == 0 and i != 299: f.write("\n    ")
        f.write("\n};\n\n")
        f.write(f"const float fc_bias = {fc_b[0]:.6f}f;\n\n#endif\n")
    print("✅ [3/3] 单片机 C 语言硬编码权重头文件已落盘: ppg_cnn_weights.h")
    print("\n🎉【大获全胜】模型组件已全部准备就绪，可以安全解耦运行验证脚本！")

if __name__ == '__main__':
    print("\n⚡ [训练引擎点火] 正在初始化纯 CPU 拓扑矩阵...")
    run_universal_save_pipeline()
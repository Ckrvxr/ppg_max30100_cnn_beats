import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import random
import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, auc

WINDOW_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

class UltimatePpgDataset(Dataset):
    def __init__(self, csv_path, window_size=100, augment=True):
        print(f"📂 正在载入已标记数据集: {csv_path} ...")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 找不到文件: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if 'IR' not in df.columns or 'RED' not in df.columns:
            raise ValueError("❌ 训练集必须包含 'IR' 和 'RED' 列！")

        ir_raw = df['IR'].values.astype(np.float64)
        red_raw = df['RED'].values.astype(np.float64)

        self.ir_hp = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values
        self.red_hp = red_raw - pd.Series(red_raw).ewm(alpha=0.04).mean().values
        self.beat_event = df['beat_event'].values
        self.window_size = window_size
        self.augment = augment
        self.num_samples = len(df) - window_size
        print(f"📊 切片数：{self.num_samples} 帧")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = idx
        if self.augment and random.random() > 0.5:
            shift = random.randint(-2, 2)
            if 0 <= idx + shift < self.num_samples:
                actual_idx = idx + shift

        x_ir = self.ir_hp[actual_idx : actual_idx + self.window_size]
        x_red = self.red_hp[actual_idx : actual_idx + self.window_size]

        combined = np.concatenate([x_ir, x_red])
        mean = combined.mean()
        std = combined.std() + 1e-6
        x_ir = (x_ir - mean) / std
        x_red = (x_red - mean) / std

        x_tensor = np.stack([x_ir, x_red], axis=0)
        half = 15
        center = actual_idx + self.window_size // 2
        s = max(0, center - half)
        e = min(len(self.beat_event), center + half + 1)
        y_target = self.beat_event[s:e].max()

        return torch.tensor(x_tensor, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

class McuPpgNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 8, kernel_size=7, padding=3, groups=2)
        self.bn1   = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2, groups=4)
        self.bn2   = nn.BatchNorm1d(16)
        self.pool1 = nn.AvgPool1d(2)
        self.pool2 = nn.AvgPool1d(2)
        self.dw3   = nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16)
        self.bn3   = nn.BatchNorm1d(16)
        self.pw3   = nn.Conv1d(16, 8, kernel_size=1)
        self.bn_pw3= nn.BatchNorm1d(8)
        self.pool3 = nn.AvgPool1d(5)
        self.dw4   = nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8)
        self.bn4   = nn.BatchNorm1d(8)
        self.pw4   = nn.Conv1d(8, 4, kernel_size=1)
        self.bn_pw4= nn.BatchNorm1d(4)
        self.fc1   = nn.Linear(4 * 5, 8)
        self.fc2   = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.dw3(x)))
        x = torch.relu(self.bn_pw3(self.pw3(x)))
        x = self.pool3(x)
        x = torch.relu(self.bn4(self.dw4(x)))
        x = torch.relu(self.bn_pw4(self.pw4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

def fuse_conv_bn(conv, bn):
    with torch.no_grad():
        w = conv.weight
        mean = bn.running_mean
        var  = bn.running_var
        gamma = bn.weight
        beta  = bn.bias
        eps   = bn.eps
        std = torch.sqrt(var + eps)
        t = gamma / std
        w_fused = w * t.view(-1, 1, 1)
        b_fused = (conv.bias - mean) * t + beta
        return w_fused.cpu().numpy(), b_fused.cpu().numpy()

def calc_metrics(y_true, y_prob):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, prec)

    preds = (y_prob >= 0.5).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0.0

    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return {
        'sensitivity': sensitivity, 'ppv': ppv, 'f1': f1, 'aupr': aupr,
        'best_threshold': best_threshold,
        'best_f1': f1_scores[best_idx],
        'best_sensitivity': rec[best_idx],
        'best_ppv': prec[best_idx],
    }

def run_universal_save_pipeline():
    train_csv = 'labeled/labeled_data.csv'
    dataset = UltimatePpgDataset(train_csv, window_size=WINDOW_SIZE, augment=True)

    # 提取标签（与 __getitem__ 对齐：窗口中心 ±15 内有 beat→正）
    labels = np.array([
        dataset.beat_event[
            max(0, i + WINDOW_SIZE // 2 - 15) :
            min(len(dataset.beat_event), i + WINDOW_SIZE // 2 + 16)
        ].max()
        for i in range(len(dataset))
    ])
    num_neg = int((labels == 0).sum())
    num_pos = int((labels == 1).sum())
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(dataset)), labels))
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f"📊 总样本={len(dataset)}  训练={len(train_idx)}  验证={len(val_idx)}")
    print(f"⚖️  负={num_neg}  正={num_pos}  权重={pos_weight:.1f}")

    # 过采样：每轮 8000 样本，正样本权重 = pos_weight
    train_labels = labels[train_idx]
    sample_weights = np.ones(len(train_idx), dtype=np.float64)
    sample_weights[train_labels == 1] = pos_weight
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=8000,
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cpu')
    print(f"📡 纯 CPU 训练 | 每轮 8000 样本过采样")

    model = McuPpgNet().to(device)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("🔥 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        avg_loss = running_loss / 8000
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}  Acc: {(correct/total)*100.0:.2f}%")

    # 验证
    print("\n══════════════════ 验证集评估 ══════════════════")
    model.eval()
    all_y = []
    all_prob = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_y.append(batch_y.numpy())
            all_prob.append(outputs.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_prob = np.concatenate(all_prob)

    m = calc_metrics(all_y, all_prob)
    print(f"正样本召回率 (Sensitivity) : {m['sensitivity']*100:.1f}%   (目标 > 90%)")
    print(f"正样本精确率 (PPV)        : {m['ppv']*100:.1f}%   (目标 > 75%)")
    print(f"F1-Score @0.5             : {m['f1']:.3f}")
    print(f"AUC-PR (PR曲线下面积)      : {m['aupr']:.3f}")
    print(f"最佳阈值 (max F1)         : {m['best_threshold']:.3f}")
    print(f"  → F1={m['best_f1']:.3f}  Sen={m['best_sensitivity']*100:.1f}%  PPV={m['best_ppv']*100:.1f}%")
    print("════════════════════════════════════════════════\n")

    # 保存模型
    print("🏆 保存模型...")
    pth_path = 'model/ppg_mcu_model.pth'
    onnx_path = 'model/ppg_mcu_model.onnx'
    h_path = 'model/ppg_cnn_weights.h'

    torch.save(model.state_dict(), pth_path)
    print(f"✅ [1/3] {pth_path}")

    model.eval()
    dummy_input = torch.randn(1, 2, WINDOW_SIZE).to(device)
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path, export_params=True, opset_version=11,
            do_constant_folding=True, input_names=['input_ppg'], output_names=['output_prob'],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
    except Exception:
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=9,
                         input_names=['input_ppg'], output_names=['output_prob'])
    print(f"✅ [2/3] {onnx_path}")

    # C 头文件（融合 BN 后导出，MCU 端无需 BN）
    model.eval()
    w1, b1 = fuse_conv_bn(model.conv1, model.bn1)
    w2, b2 = fuse_conv_bn(model.conv2, model.bn2)
    w3, b3 = fuse_conv_bn(model.dw3, model.bn3)
    w4, b4 = fuse_conv_bn(model.pw3, model.bn_pw3)
    w5, b5 = fuse_conv_bn(model.dw4, model.bn4)
    w6, b6 = fuse_conv_bn(model.pw4, model.bn_pw4)
    fc1_w = model.fc1.weight.detach().numpy()
    fc1_b = model.fc1.bias.detach().numpy()
    fc2_w = model.fc2.weight.detach().numpy()
    fc2_b = model.fc2.bias.detach().numpy()

    with open(h_path, 'w', encoding='utf-8') as f:
        f.write("// [Embedded TinyML] 1D-CNN IR+RED Pulse Detection Weights\n")
        f.write(f"#define CNN_WINDOW_SIZE {WINDOW_SIZE}\n\n")

        def write_conv(w, b, name):
            f.write(f"// {name} [{w.shape[0]}][{w.shape[1]}][{w.shape[2]}]\n")
            f.write(f"const float {name}_weight[{w.shape[0]}][{w.shape[1]}][{w.shape[2]}] = {{\n")
            for oc in range(w.shape[0]):
                f.write("    {\n")
                for ic in range(w.shape[1]):
                    ws = ", ".join(f"{x:.6f}f" for x in w[oc][ic])
                    f.write(f"        {{{ws}}},\n")
                f.write("    },\n")
            f.write("};\n")
            bs = ", ".join(f"{x:.6f}f" for x in b)
            f.write(f"const float {name}_bias[{w.shape[0]}] = {{ {bs} }};\n\n")

        write_conv(w1, b1, "conv1")
        write_conv(w2, b2, "conv2")
        write_conv(w3, b3, "dw3")
        write_conv(w4, b4, "pw3")
        write_conv(w5, b5, "dw4")
        write_conv(w6, b6, "pw4")

        f.write(f"const float fc1_weight[{fc1_w.shape[0]}][{fc1_w.shape[1]}] = {{\n")
        for i in range(fc1_w.shape[0]):
            ws = ", ".join(f"{x:.6f}f" for x in fc1_w[i])
            f.write(f"    {{{ws}}},\n")
        f.write("};\n")
        bs = ", ".join(f"{x:.6f}f" for x in fc1_b)
        f.write(f"const float fc1_bias[{fc1_b.shape[0]}] = {{ {bs} }};\n\n")

        ws = ", ".join(f"{x:.6f}f" for x in fc2_w[0])
        f.write(f"const float fc2_weight[{fc2_w.shape[1]}] = {{ {ws} }};\n")
        f.write(f"const float fc2_bias = {fc2_b[0]:.6f}f;\n\n#endif\n")
    print(f"✅ [3/3] {h_path}")
    print("\n🎉 全部完成！")

if __name__ == '__main__':
    print("\n⚡ 初始化训练...")
    run_universal_save_pipeline()

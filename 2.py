import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import random
import os
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
        x_ir_norm = (x_ir - np.mean(x_ir)) / (np.std(x_ir) + 1e-6)

        x_red = self.red_hp[actual_idx : actual_idx + self.window_size]
        x_red_norm = (x_red - np.mean(x_red)) / (np.std(x_red) + 1e-6)

        x_tensor = np.stack([x_ir_norm, x_red_norm], axis=0)
        y_target = self.beat_event[actual_idx + self.window_size - 1]

        return torch.tensor(x_tensor, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x).squeeze(-1)

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

    labels = dataset.beat_event[WINDOW_SIZE-1:WINDOW_SIZE-1+len(dataset)]
    num_neg = (labels == 0).sum()
    num_pos = (labels == 1).sum()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    print(f"📊 总样本={len(dataset)}  训练={len(train_idx)}  验证={len(val_idx)}")
    print(f"⚖️  负={num_neg}  正={num_pos}  权重={pos_weight:.1f}")

    device = torch.device('cpu')
    print(f"📡 纯 CPU 训练")

    model = MultiChannelMcuCnn().to(device)
    criterion = nn.BCELoss(reduction='none')
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
            weights = batch_y * pos_weight + (1 - batch_y)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_dataset):.4f}  Acc: {(correct/total)*100.0:.2f}%")

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

    with torch.no_grad():
        c_weight = model.conv.weight.numpy()
        c_bias = model.conv.bias.numpy()
        fc_w = model.fc.weight.numpy()
        fc_b = model.fc.bias.numpy()

    with open(h_path, 'w', encoding='utf-8') as f:
        f.write("// [Embedded TinyML] 1D-CNN IR+RED Pulse Detection Weights\n")
        f.write(f"#define CNN_WINDOW_SIZE {WINDOW_SIZE}\n\n")
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
            if (i + 1) % 10 == 0 and i != 299:
                f.write("\n    ")
        f.write("\n};\n\n")
        f.write(f"const float fc_bias = {fc_b[0]:.6f}f;\n\n#endif\n")
    print(f"✅ [3/3] {h_path}")
    print("\n🎉 全部完成！")

if __name__ == '__main__':
    print("\n⚡ 初始化训练...")
    run_universal_save_pipeline()

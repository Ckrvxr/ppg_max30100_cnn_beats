import argparse
import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from io import StringIO
from scipy.signal import medfilt


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


def kalman_filter(probs, Q=0.01, R=0.1):
    n = len(probs)
    x = np.zeros(n)
    P = np.zeros(n)
    x[0] = probs[0]
    P[0] = 1.0
    for i in range(1, n):
        x_pred = x[i-1]
        P_pred = P[i-1] + Q
        K = P_pred / (P_pred + R)
        x[i] = x_pred + K * (probs[i] - x_pred)
        P[i] = (1 - K) * P_pred
    return x


def run_prelabel(input_path, output_csv, model_path, window_size, threshold, refractory, kalman_Q=0.01, kalman_R=0.1):
    print(f"📂 载入: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在")
        return
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return

    if input_path.endswith('.csv'):
        with open(input_path) as f:
            cols = f.readline().strip().split(',')
        if len(cols) == 4 and cols[0] == 'IR' and cols[1] == 'RED':
            df = pd.read_csv(input_path, header=None, names=['a','b','IR','RED'])
            ir = df['IR'].values.astype(np.float64)
            red = df['RED'].values.astype(np.float64)
            n = len(df)
            print(f"📊 CSV: {n} 点 ({n/6000:.1f} 分钟)")
        else:
            df = pd.read_csv(input_path)
            df.columns = df.columns.str.strip()
            ir = df['IR'].values.astype(np.float64)
            red = df['RED'].values.astype(np.float64)
            n = len(df)
            print(f"📊 CSV: {n} 点, beat=1: {(df['beat_event'].sum() if 'beat_event' in df else 0)}")
    else:
        with open(input_path, 'rb') as f:
            raw = f.read()
        idx = raw.find(b'IR,RED')
        if idx == -1:
            print("❌ 找不到 IR,RED 标记")
            return
        df = pd.read_csv(StringIO(raw[idx:].decode('ascii')), header=None, names=['a','b','IR','RED'])
        ir = df['IR'].values.astype(np.float64)
        red = df['RED'].values.astype(np.float64)
        n = len(df)
        print(f"📊 TXT: {n} 点 ({n/6000:.1f} 分钟)")

    ir_hp = ir - pd.Series(ir).ewm(alpha=0.04).mean().values
    red_hp = red - pd.Series(red).ewm(alpha=0.04).mean().values

    model = McuPpgNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    print("🏎️  PyTorch 推理...")
    probs = np.zeros(n)
    for i in range(window_size, n):
        xi = ir_hp[i-window_size:i]
        xr = red_hp[i-window_size:i]
        c = np.concatenate([xi, xr])
        m, s = c.mean(), c.std() + 1e-6
        xi = (xi - m) / s
        xr = (xr - m) / s
        x_t = torch.tensor(np.stack([xi, xr], axis=0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs[i] = model(x_t).item()

    integral = medfilt(probs, kernel_size=9)
    beat = np.zeros(n, dtype=int)
    above = False
    last = -refractory
    for i in range(n):
        if integral[i] >= threshold:
            if not above and i - last >= refractory:
                beat[i] = 1
                last = i
            above = True
        else:
            above = False

    shifted = np.zeros(n, dtype=int)
    shifted[:n-30] = beat[30:]
    beat = shifted

    print(f"✅ 检测到 {beat.sum()} 个事件")
    pd.DataFrame({'IR': ir, 'RED': red, 'beat_event': beat}).to_csv(output_csv, index=False)
    print(f"💾 {output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPG 预标记工具')
    parser.add_argument('input', help='输入数据文件（.txt 或 .csv）')
    parser.add_argument('--output', '-o', help='输出 CSV 路径（默认覆盖输入文件）')
    parser.add_argument('--config', '-c', default='ppg_config.json',
                        help='配置文件路径（默认 ppg_config.json）')
    parser.add_argument('--threshold', type=float, help='覆盖检测阈值')
    args = parser.parse_args()

    with open(args.config) as f:
        CFG = json.load(f)

    output_csv = args.output if args.output else args.input
    threshold = args.threshold if args.threshold is not None else CFG['prelabel']['threshold']

    print("⚡ 预标记工具")
    run_prelabel(args.input, output_csv, CFG['model']['pth'],
                 CFG['train']['window_size'], threshold, CFG['prelabel']['refractory'])

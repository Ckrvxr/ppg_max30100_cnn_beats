import argparse
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from io import StringIO
from scipy.signal import medfilt


class McuPpgNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(2, 8, kernel_size=7, padding=3, groups=2)
        self.bn1   = torch.nn.BatchNorm1d(8)
        self.conv2 = torch.nn.Conv1d(8, 16, kernel_size=5, padding=2, groups=4)
        self.bn2   = torch.nn.BatchNorm1d(16)
        self.pool1 = torch.nn.AvgPool1d(2)
        self.pool2 = torch.nn.AvgPool1d(2)
        self.dw3   = torch.nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16)
        self.bn3   = torch.nn.BatchNorm1d(16)
        self.pw3   = torch.nn.Conv1d(16, 8, kernel_size=1)
        self.bn_pw3= torch.nn.BatchNorm1d(8)
        self.pool3 = torch.nn.AvgPool1d(5)
        self.dw4   = torch.nn.Conv1d(8, 8, kernel_size=3, padding=1, groups=8)
        self.bn4   = torch.nn.BatchNorm1d(8)
        self.pw4   = torch.nn.Conv1d(8, 4, kernel_size=1)
        self.bn_pw4= torch.nn.BatchNorm1d(4)
        self.fc1   = torch.nn.Linear(4 * 5, 8)
        self.fc2   = torch.nn.Linear(8, 1)
        self.sigmoid = torch.nn.Sigmoid()

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


def run_verification(sources, window_size, pth_path, trigger_thr, refractory=20, fs=100):
    sources = [s for s in sources if os.path.exists(s)]
    if not sources:
        print("❌ 无可用数据源")
        return

    if not os.path.exists(pth_path):
        print(f"❌ 找不到模型: {pth_path}")
        return
    model = McuPpgNet()
    model.load_state_dict(torch.load(pth_path, weights_only=True))
    model.eval()
    print(f"📡 模型已加载: {pth_path}")

    for src in sources:
        print(f"\n📂 {src}")
        if src.endswith('.csv'):
            df = pd.read_csv(src)
            df.columns = df.columns.str.strip()
        else:
            with open(src, 'rb') as f:
                raw = f.read()
            idx = raw.find(b'IR,RED')
            if idx == -1:
                print("  ❌ 无 IR,RED 标记, 跳过")
                continue
            df = pd.read_csv(StringIO(raw[idx:].decode('ascii')), header=None, names=['a','b','IR','RED'])
        ir = df['IR'].values.astype(np.float64)
        red = df['RED'].values.astype(np.float64)
        n = len(df)

        ir_hp = ir - pd.Series(ir).ewm(alpha=0.04).mean().values
        red_hp = red - pd.Series(red).ewm(alpha=0.04).mean().values

        print(f"  🏎️  PyTorch 推理 {n} 点...")
        probs = np.zeros(n)
        batch_size = 1024
        with torch.no_grad():
            for st in range(window_size, n, batch_size):
                en = min(st + batch_size, n)
                batch = []
                for i in range(st, en):
                    xi = ir_hp[i-window_size:i]
                    xr = red_hp[i-window_size:i]
                    c = np.concatenate([xi, xr])
                    m, s = c.mean(), c.std() + 1e-6
                    xi = (xi - m) / s
                    xr = (xr - m) / s
                    batch.append(np.stack([xi, xr], axis=0))
                batch_t = torch.from_numpy(np.array(batch, dtype=np.float32))
                probs[st:en] = model(batch_t).numpy()

        kernel = np.ones(5) / 5
        probs_ma  = np.convolve(probs, kernel, mode='same')
        probs_ewm = pd.Series(probs).ewm(alpha=0.3).mean().values
        probs_med = medfilt(probs, kernel_size=13)
        probs_int = np.convolve(probs, np.ones(10) / 10, mode='same')

        beats = np.zeros(n, dtype=int)
        above = False
        last_beat = -refractory
        for i in range(n):
            if probs_med[i] >= trigger_thr:
                if not above and i - last_beat >= refractory:
                    beats[i] = 1
                    last_beat = i
                above = True
            else:
                above = False

        beat_idx = np.where(beats == 1)[0]
        bpm_inst = np.zeros(n)
        if len(beat_idx) > 3:
            window = []
            for i in range(len(beat_idx)):
                if window and beat_idx[i] - window[-1] > 158:
                    window = []
                window.append(beat_idx[i])
                if len(window) == 4:
                    ppi = np.diff(window) / fs
                    bpm = 60.0 / np.mean(ppi)
                    st = window[1]
                    en = window[2]
                    bpm_inst[st:en] = bpm
                    window = window[1:]

        bpm_hold = bpm_inst.copy()
        last_valid = 0
        zero_count = 0
        hold_samples = int(15 * fs)
        for i in range(n):
            if bpm_inst[i] > 0:
                last_valid = bpm_inst[i]
                zero_count = 0
            else:
                zero_count += 1
                if zero_count < hold_samples:
                    bpm_hold[i] = last_valid

        bpm_smooth = pd.Series(bpm_hold).ewm(alpha=0.04).mean().values

        CHUNK = 3000
        cur = 0
        while cur < n:
            end = min(cur + CHUNK, n)
            xrng = range(cur, end)
            fig, axes = plt.subplots(7, 1, sharex=True, figsize=(15, 13))
            fig.suptitle(f"{os.path.basename(src)}  [{cur}~{end} / {n}]", fontsize=13, fontweight='bold')
            aw, a1, a2, a3, a4, a5, a6 = axes

            aw.plot(xrng, ir_hp[cur:end], color='#2ec4b6', linewidth=1.2)
            aw.set_ylim(-2000, 2000); aw.set_ylabel('① IR EWM'); aw.grid(True, ls='--', alpha=0.5)

            a1.plot(xrng, probs[cur:end], color='#ff1654', linewidth=0.8)
            a1.axhline(y=trigger_thr, color='gray', ls=':', alpha=0.5)
            a1.set_ylim(-0.05, 1.05)
            a1.set_ylabel('② RAW Confidence', fontweight='bold', color='#ff1654')
            a1.grid(True, ls='--', alpha=0.5)

            a2.plot(xrng, probs_med[cur:end], color='#FF9800', linewidth=1.2)
            a2.axhline(y=trigger_thr, color='gray', ls=':', alpha=0.5)
            a2.set_ylim(-0.05, 1.05)
            a2.set_ylabel('③ Medfilt13', fontweight='bold', color='#FF9800')
            a2.grid(True, ls='--', alpha=0.5)

            a3.vlines(xrng, ymin=0, ymax=beats[cur:end], color='red', linewidth=1.5, alpha=0.8)
            a3.set_ylim(-0.1, 1.1)
            a3.set_ylabel('④ Detected Beats', fontweight='bold', color='red')
            a3.grid(True, ls='--', alpha=0.5)

            chunk_beats = [b for b in beat_idx if cur <= b < end]
            aw.vlines(chunk_beats, ymin=-2000, ymax=2000, color='red', linewidth=0.8, alpha=0.6)

            a4.set_xlabel('Samples Index')
            a4.plot(xrng, bpm_inst[cur:end], color='#2196F3', linewidth=1.2, alpha=0.8)
            a4.set_ylim(0, 210)
            a4.set_ylabel('⑤ BPM', fontweight='bold', color='#2196F3')
            a4.axhline(y=72, color='gray', ls=':', alpha=0.5)
            a4.grid(True, ls='--', alpha=0.5)
            for b in chunk_beats:
                val = bpm_inst[b]
                if val > 0:
                    a4.text(b, val + 4, f'{val:.0f}', fontsize=7, ha='center', va='bottom',
                            color='#2196F3', fontweight='bold')

            a5.set_xlabel('Samples Index')
            a5.plot(xrng, bpm_hold[cur:end], color='#4CAF50', linewidth=1.2, alpha=0.8)
            a5.set_ylim(0, 210)
            a5.set_ylabel('⑥ BPM (hold 15s)', fontweight='bold', color='#4CAF50')
            a5.axhline(y=72, color='gray', ls=':', alpha=0.5)
            a5.grid(True, ls='--', alpha=0.5)
            for b in chunk_beats:
                val = bpm_hold[b]
                if val > 0:
                    a5.text(b, val + 4, f'{val:.0f}', fontsize=7, ha='center', va='bottom',
                            color='#4CAF50', fontweight='bold')

            a6.set_xlabel('Samples Index')
            a6.plot(xrng, bpm_smooth[cur:end], color='#E91E63', linewidth=1.2, alpha=0.8)
            a6.set_ylim(0, 210)
            a6.set_ylabel('⑦ EWM Smooth', fontweight='bold', color='#E91E63')
            a6.axhline(y=72, color='gray', ls=':', alpha=0.5)
            a6.grid(True, ls='--', alpha=0.5)
            for b in chunk_beats:
                val = bpm_smooth[b]
                if val > 0:
                    a6.text(b, val + 4, f'{val:.0f}', fontsize=7, ha='center', va='bottom',
                            color='#E91E63', fontweight='bold')

            def mk_cb():
                nonlocal cur
                def cb(ev):
                    nonlocal cur
                    if ev.key == 'right':
                        cur += CHUNK; plt.close(fig)
                    elif ev.key == 'left' and cur >= CHUNK:
                        cur -= CHUNK; plt.close(fig)
                    elif ev.key == 'escape':
                        cur = n; plt.close(fig)
                return cb

            fig.canvas.mpl_connect('key_press_event', mk_cb())
            plt.tight_layout()
            plt.show()

    print("\n✅ 验证完成")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPG 模型验证与可视化')
    parser.add_argument('--config', '-c', default='ppg_config.json',
                        help='配置文件路径（默认 ppg_config.json）')
    parser.add_argument('--source', nargs='+', help='覆盖数据源列表')
    args = parser.parse_args()

    with open(args.config) as f:
        CFG = json.load(f)

    sources = args.source if args.source else CFG['data']['sources']
    window_size = CFG['train']['window_size']
    pth_path = CFG['model']['pth']
    trigger_thr = CFG.get('prelabel', {}).get('threshold', 0.5)
    refractory = CFG.get('prelabel', {}).get('refractory', 20)
    fs = CFG.get('sampling_rate', 100)
    print("⚡ 验证启动")
    run_verification(sources, window_size, pth_path, trigger_thr, refractory, fs)

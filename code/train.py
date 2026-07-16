import argparse
import json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, auc
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


def prelabel_txt(txt_path, out_csv, model_path, window_size, threshold, refractory):
    with open(txt_path, 'rb') as f:
        raw = f.read()
    idx = raw.find(b'IR,RED')
    if idx == -1:
        return False
    csv_part = raw[idx:].decode('ascii')
    df = pd.read_csv(StringIO(csv_part), header=None, names=['tag1','tag2','IR','RED'])
    ir_raw = df['IR'].values.astype(np.float64)
    red_raw = df['RED'].values.astype(np.float64)
    ir_hp = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values
    red_hp = red_raw - pd.Series(red_raw).ewm(alpha=0.04).mean().values
    total = len(df)
    model = McuPpgNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    probs = np.zeros(total)
    for i in range(window_size, total):
        xi = ir_hp[i-window_size:i]
        xr = red_hp[i-window_size:i]
        c = np.concatenate([xi, xr])
        m, s = c.mean(), c.std() + 1e-6
        xi = (xi - m) / s
        xr = (xr - m) / s
        x_t = torch.tensor(np.stack([xi, xr], axis=0), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs[i] = model(x_t).item()
    smoothed = medfilt(probs, kernel_size=9)
    beat = np.zeros(total, dtype=int)
    above = False
    last = -refractory
    for i in range(total):
        if smoothed[i] >= threshold:
            if not above and i - last >= refractory:
                beat[i] = 1
                last = i
            above = True
        else:
            above = False
    shifted = np.zeros(total, dtype=int)
    shifted[:total-30] = beat[30:]
    beat = shifted
    pd.DataFrame({'IR': ir_raw, 'RED': red_raw, 'beat_event': beat}).to_csv(out_csv, index=False)
    print(f"  ✅ 预标记 {len(beat)} 点, beat=1: {beat.sum()}")
    return True


def merge_dataset(cfg):
    print("📂 合并数据源...")
    all_dfs = []
    for src in cfg['data']['sources']:
        if not os.path.exists(src):
            print(f"  ⚠️ 跳过: {src}")
            continue
        if src.endswith('.csv'):
            print(f"  📂 读取已标记: {src}")
            df = pd.read_csv(src)
        else:
            base = os.path.splitext(os.path.basename(src))[0]
            label_path = os.path.join('labeled', base + '_labeled.csv')
            if not os.path.exists(label_path):
                print(f"  🏎️  预标记: {src}")
                if not prelabel_txt(src, label_path, cfg['model']['pth'],
                                    cfg['train']['window_size'], cfg['prelabel']['threshold'],
                                    cfg['prelabel']['refractory']):
                    print(f"  ❌ 预标记失败: {src}")
                    continue
            df = pd.read_csv(label_path)
        if 'beat_event' not in df.columns:
            df['beat_event'] = 0
        all_dfs.append(df)
        print(f"  📊 {len(df)} 行, beat=1: {(df.beat_event==1).sum()}")
    if not all_dfs:
        raise RuntimeError("❌ 无有效数据源")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(cfg['data']['dataset'], index=False)
    print(f"  🚀 合并完成: {len(combined)} 行, beat=1: {(combined.beat_event==1).sum()}")
    return combined


class UltimatePpgDataset(Dataset):
    def __init__(self, df, window_size, augment=True):
        ir_raw = df['IR'].values.astype(np.float64)
        red_raw = df['RED'].values.astype(np.float64)
        self.ir_hp = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values
        self.red_hp = red_raw - pd.Series(red_raw).ewm(alpha=0.04).mean().values
        self.beat_event = df['beat_event'].values
        self.window_size = window_size
        self.augment = augment
        self.num_samples = len(df) - window_size
        print(f"📊 切片数: {self.num_samples} 帧")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.augment and random.random() > 0.5:
            shift = random.randint(-2, 2)
            if 0 <= idx + shift < self.num_samples:
                idx = idx + shift
        xi = self.ir_hp[idx : idx + self.window_size]
        xr = self.red_hp[idx : idx + self.window_size]
        c = np.concatenate([xi, xr])
        m, s = c.mean(), c.std() + 1e-6
        xi = (xi - m) / s
        xr = (xr - m) / s
        center = idx + self.window_size // 2
        a = max(0, center - 5)
        b = min(len(self.beat_event), center + 5)
        return (torch.tensor(np.stack([xi, xr], axis=0), dtype=torch.float32),
                torch.tensor(self.beat_event[a:b].max(), dtype=torch.float32))


def fuse_conv_bn(conv, bn):
    with torch.no_grad():
        w = conv.weight
        m, v = bn.running_mean, bn.running_var
        g, b, e = bn.weight, bn.bias, bn.eps
        t = g / torch.sqrt(v + e)
        return (w * t.view(-1, 1, 1)).cpu().numpy(), ((conv.bias - m) * t + b).cpu().numpy()


def calc_metrics(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    aupr = auc(rec, prec)
    preds = (y_prob >= 0.5).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()
    sn = tp / (tp + fn) if tp + fn > 0 else 0
    pp = tp / (tp + fp) if tp + fp > 0 else 0
    f1 = 2 * sn * pp / (sn + pp) if sn + pp > 0 else 0
    fs = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    bi = np.argmax(fs)
    return {'sensitivity': sn, 'ppv': pp, 'f1': f1, 'aupr': aupr,
            'best_threshold': thr[bi] if len(thr) > bi else 0.5,
            'best_f1': fs[bi], 'best_sensitivity': rec[bi], 'best_ppv': prec[bi]}


def run_training(cfg):
    df = merge_dataset(cfg)
    window_size = cfg['train']['window_size']
    dataset = UltimatePpgDataset(df, window_size=window_size, augment=True)

    labels = np.array([dataset.beat_event[
        max(0, i + window_size // 2 - 5) : min(len(dataset.beat_event), i + window_size // 2 + 5)
    ].max() for i in range(len(dataset))])
    num_neg = int((labels == 0).sum())
    num_pos = int((labels == 1).sum())
    pw = num_neg / num_pos if num_pos > 0 else 1.0

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tidx, vidx = next(sss.split(np.zeros(len(dataset)), labels))
    t_sub = Subset(dataset, tidx)
    v_sub = Subset(dataset, vidx)

    print(f"📊 训练={len(tidx)} 验证={len(vidx)}  负={num_neg} 正={num_pos} 权重={pw:.1f}")

    sw = np.ones(len(tidx), dtype=np.float64)
    sw[labels[tidx] == 1] = pw
    t_loader = DataLoader(t_sub, batch_size=cfg['train']['batch_size'],
                          sampler=WeightedRandomSampler(sw.tolist(), cfg['train']['num_samples_epoch'], True))
    v_loader = DataLoader(v_sub, batch_size=cfg['train']['batch_size'], shuffle=False)

    device = torch.device('cpu')
    model = McuPpgNet().to(device)
    cri = nn.BCELoss(reduction='mean')
    opt = optim.AdamW(model.parameters(), lr=cfg['train']['learning_rate'], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])

    print("🔥 训练...")
    for ep in range(cfg['train']['epochs']):
        model.train()
        loss_sum = 0
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = cri(model(bx), by)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * bx.size(0)
        sch.step()
        print(f"  Epoch {ep+1}/{cfg['train']['epochs']} Loss: {loss_sum/cfg['train']['num_samples_epoch']:.4f}")

    model.eval()
    ay, ap = [], []
    with torch.no_grad():
        for bx, by in v_loader:
            ay.append(by.numpy())
            ap.append(model(bx.to(device)).cpu().numpy())
    ay = np.concatenate(ay)
    ap = np.concatenate(ap)
    m = calc_metrics(ay, ap)
    print("\n══════════════════ 验证集 ══════════════════")
    print(f"  Sen: {m['sensitivity']*100:.1f}%  PPV: {m['ppv']*100:.1f}%  F1: {m['f1']:.3f}  AUC-PR: {m['aupr']:.3f}")
    print(f"  最佳阈值: {m['best_threshold']:.3f} (F1={m['best_f1']:.3f} Sen={m['best_sensitivity']*100:.1f}% PPV={m['best_ppv']*100:.1f}%)")
    print("══════════════════════════════════════════════\n")

    torch.save(model.state_dict(), cfg['model']['pth'])
    print(f"✅ [1/3] {cfg['model']['pth']}")

    dummy = torch.randn(1, 2, cfg['train']['window_size'])
    torch.onnx.export(model, dummy, cfg['model']['onnx'],
                      input_names=['input'], output_names=['output'])
    print(f"✅ [2/3] {cfg['model']['onnx']}")

    me = model.eval()
    w1, b1 = fuse_conv_bn(me.conv1, me.bn1)
    w2, b2 = fuse_conv_bn(me.conv2, me.bn2)
    w3, b3 = fuse_conv_bn(me.dw3, me.bn3)
    w4, b4 = fuse_conv_bn(me.pw3, me.bn_pw3)
    w5, b5 = fuse_conv_bn(me.dw4, me.bn4)
    w6, b6 = fuse_conv_bn(me.pw4, me.bn_pw4)
    f1w = me.fc1.weight.detach().numpy()
    f1b = me.fc1.bias.detach().numpy()
    f2w = me.fc2.weight.detach().numpy()
    f2b = me.fc2.bias.detach().numpy()

    with open(cfg['model']['h'], 'w') as f:
        f.write(f"#define CNN_WINDOW_SIZE {window_size}\n\n")
        def wc(w, b, n):
            f.write(f"const float {n}_weight[{w.shape[0]}][{w.shape[1]}][{w.shape[2]}] = {{\n")
            for oc in range(w.shape[0]):
                f.write("    {\n")
                for ic in range(w.shape[1]):
                    f.write(f"        {{{', '.join(f'{x:.6f}f' for x in w[oc][ic])}}},\n")
                f.write("    },\n")
            f.write("};\n")
            f.write(f"const float {n}_bias[{w.shape[0]}] = {{{', '.join(f'{x:.6f}f' for x in b)}}};\n\n")
        wc(w1, b1, "conv1"); wc(w2, b2, "conv2"); wc(w3, b3, "dw3")
        wc(w4, b4, "pw3"); wc(w5, b5, "dw4"); wc(w6, b6, "pw4")
        f.write(f"const float fc1_weight[{f1w.shape[0]}][{f1w.shape[1]}] = {{\n")
        for r in range(f1w.shape[0]):
            f.write(f"    {{{', '.join(f'{x:.6f}f' for x in f1w[r])}}},\n")
        f.write("};\n")
        f.write(f"const float fc1_bias[{f1b.shape[0]}] = {{{', '.join(f'{x:.6f}f' for x in f1b)}}};\n\n")
        f.write(f"const float fc2_weight[{f2w.shape[1]}] = {{{', '.join(f'{x:.6f}f' for x in f2w[0])}}};\n")
        f.write(f"const float fc2_bias = {f2b[0]:.6f}f;\n\n#endif\n")
    print(f"✅ [3/3] {cfg['model']['h']}")
    print("\n🎉 全部完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPG CNN 模型训练')
    parser.add_argument('--config', '-c', default='ppg_config.json',
                        help='配置文件路径（默认 ppg_config.json）')
    parser.add_argument('--epochs', type=int, help='覆盖训练轮数')
    parser.add_argument('--lr', type=float, help='覆盖学习率')
    parser.add_argument('--batch-size', type=int, help='覆盖 batch size')
    args = parser.parse_args()

    with open(args.config) as f:
        CFG = json.load(f)

    if args.epochs is not None:
        CFG['train']['epochs'] = args.epochs
    if args.lr is not None:
        CFG['train']['learning_rate'] = args.lr
    if args.batch_size is not None:
        CFG['train']['batch_size'] = args.batch_size

    print("⚡ 训练启动")
    run_training(CFG)

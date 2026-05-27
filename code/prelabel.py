import argparse
import json, os
import numpy as np
import pandas as pd
import onnxruntime as ort
from io import StringIO


def run_prelabel(input_path, output_csv, onnx_path, window_size, threshold, refractory):
    print(f"📂 载入: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在")
        return
    if not os.path.exists(onnx_path):
        print(f"❌ 模型不存在: {onnx_path}")
        return

    if input_path.endswith('.csv'):
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

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    print("🏎️  ONNX 推理...")
    probs = np.zeros(n)
    for i in range(window_size, n):
        xi = ir_hp[i-window_size:i]
        xr = red_hp[i-window_size:i]
        c = np.concatenate([xi, xr])
        m, s = c.mean(), c.std() + 1e-6
        xi = (xi - m) / s
        xr = (xr - m) / s
        x_t = np.stack([xi, xr], axis=0).reshape(1, 2, window_size).astype(np.float32)
        probs[i] = sess.run([oname], {iname: x_t})[0].item()

    integral = np.convolve(probs, np.ones(20) / 20, mode='same')
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
    parser = argparse.ArgumentParser(description='PPG ONNX 预标记工具')
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
    run_prelabel(args.input, output_csv, CFG['model']['onnx'],
                 CFG['train']['window_size'], threshold, CFG['prelabel']['refractory'])

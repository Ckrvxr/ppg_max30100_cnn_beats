import json, os, sys
import numpy as np
import pandas as pd
import onnxruntime as ort
from io import StringIO

with open('ppg_config.json') as f:
    CFG = json.load(f)

ONNX_PATH   = CFG['model']['onnx']
WINDOW_SIZE = CFG['train']['window_size']
THR         = CFG['prelabel']['threshold']
REFRACTORY  = CFG['prelabel']['refractory']

def prelabel(input_path, output_csv):
    print(f"📂 载入: {input_path}")
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在")
        return
    if not os.path.exists(ONNX_PATH):
        print(f"❌ 模型不存在: {ONNX_PATH}")
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

    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name

    print("🏎️  ONNX 推理...")
    probs = np.zeros(n)
    for i in range(WINDOW_SIZE, n):
        xi = ir_hp[i-WINDOW_SIZE:i]
        xr = red_hp[i-WINDOW_SIZE:i]
        c = np.concatenate([xi, xr])
        m, s = c.mean(), c.std() + 1e-6
        xi = (xi - m) / s
        xr = (xr - m) / s
        x_t = np.stack([xi, xr], axis=0).reshape(1, 2, WINDOW_SIZE).astype(np.float32)
        probs[i] = sess.run([oname], {iname: x_t})[0].item()

    integral = np.convolve(probs, np.ones(20) / 20, mode='same')
    beat = np.zeros(n, dtype=int)
    above = False
    last = -REFRACTORY
    for i in range(n):
        if integral[i] >= THR:
            if not above and i - last >= REFRACTORY:
                beat[i] = 1
                last = i
            above = True
        else:
            above = False

    # 左移 30 点对齐波峰位置
    shifted = np.zeros(n, dtype=int)
    shifted[:n-30] = beat[30:]
    beat = shifted

    print(f"✅ 检测到 {beat.sum()} 个事件")
    pd.DataFrame({'IR': ir, 'RED': red, 'beat_event': beat}).to_csv(output_csv, index=False)
    print(f"💾 {output_csv}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: pixi run python 4.py <input.txt/csv> [output.csv]")
        print("  .txt → 解析后预标记")
        print("  .csv → 读取 IR,RED 重新预标记")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else inp
    print("⚡ 预标记工具")
    prelabel(inp, out)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
import os
import sys
from io import StringIO

CNN_WINDOW_SIZE = 100
ONNX_MODEL_PATH = 'model/ppg_mcu_model.onnx'
TEST_DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else 'raw/raw_data.txt'

def run_onnx_cpu_verification():
    print("\n⚡ [ONNX CPU 验证器] 启动...")

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 找不到模型: {ONNX_MODEL_PATH}")
        return
    print(f"📡 载入模型: {ONNX_MODEL_PATH} ...")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 找不到数据: {TEST_DATA_PATH}")
        return

    print(f"📂 载入数据: {TEST_DATA_PATH} ...")
    with open(TEST_DATA_PATH, 'rb') as f:
        raw = f.read()
    idx = raw.find(b'IR,RED')
    if idx == -1:
        print("❌ 找不到 'IR,RED' 标记")
        return
    csv_part = raw[idx:].decode('ascii')
    df = pd.read_csv(StringIO(csv_part), header=None, names=['tag1', 'tag2', 'IR', 'RED'])

    ir_raw = df['IR'].values.astype(np.float64)
    red_raw = df['RED'].values.astype(np.float64)
    total_len = len(df)
    print(f"📊 {total_len} 点 ({total_len/6000:.1f} 分钟)")

    ir_hp = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values
    red_hp = red_raw - pd.Series(red_raw).ewm(alpha=0.04).mean().values

    print("🏎️  滚动滑窗推理...")
    probs = np.zeros(total_len)
    for i in range(CNN_WINDOW_SIZE, total_len):
        x_ir = ir_hp[i - CNN_WINDOW_SIZE : i]
        x_red = red_hp[i - CNN_WINDOW_SIZE : i]
        # 联合归一化（与 2.py 训练对齐）
        combined = np.concatenate([x_ir, x_red])
        mean = combined.mean()
        std = combined.std() + 1e-6
        x_ir = (x_ir - mean) / std
        x_red = (x_red - mean) / std
        x_input = np.stack([x_ir, x_red], axis=0)
        x_tensor = np.expand_dims(x_input, axis=0).astype(np.float32)
        probs[i] = session.run([output_name], {input_name: x_tensor})[0].item()

    print("🏆 推理完成！渲染可视化...\n")

    CHUNK = 5000
    n_chunks = int(np.ceil(total_len / CHUNK))

    for chunk in range(n_chunks):
        start = chunk * CHUNK
        end = min(start + CHUNK, total_len)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
        fig.suptitle(f"Verification Chunk {chunk+1}/{n_chunks}  [{start} ~ {end}]", fontsize=13, fontweight='bold')

        ax1.plot(range(start, end), ir_hp[start:end], color='#2ec4b6', linewidth=1.2)
        ax1.set_ylim(-2000, 2000)
        ax1.set_ylabel('IR EWM', fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2.plot(range(start, end), probs[start:end], color='#ff1654', linewidth=1.5)
        ax2.axhline(y=0.82, color='gray', linestyle=':', label='Threshold (0.82)')
        ax2.set_ylabel('Confidence', fontweight='bold')
        ax2.set_xlabel('Samples Index', fontweight='bold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    print("✅ 验证完成")

if __name__ == '__main__':
    run_onnx_cpu_verification()

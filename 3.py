import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort
import os

# ==============================================================================
# 🎛️ 工业标准·纯 CPU 验证配置
# ==============================================================================
CNN_WINDOW_SIZE = 100
ONNX_MODEL_PATH = 'ppg_mcu_model.onnx'      # 读取 2.py 生成的通用 ONNX 文件
TEST_DATA_PATH = '2026-05-27_02-14-03.csv'  # 待验证的全新未标注 CSV 文件

def run_onnx_cpu_verification():
    print("\n⚡ [ONNX CPU 仿真验证器点火] 正在初始化轻量级通用计算链路...")
    
    # 1. 🛡️ 严格审查 ONNX 实体是否存在
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ 错误：在当前目录下找不到 '{ONNX_MODEL_PATH}'！请先执行 'pixi run python 2.py' 进行训练生成。")
        return
        
    print(f"📡 正在拉起工业级开放推理引擎加载: {ONNX_MODEL_PATH} ...")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # 自动对齐图结构的节点别名
    input_name = session.get_inputs()[0].name   # 'input_ppg'
    output_name = session.get_outputs()[0].name # 'output_prob'
    
    # 2. 📂 载入并清洗测试集
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ 错误：找不到测试数据集文件 '{TEST_DATA_PATH}'！")
        return
        
    print(f"📂 正在载入待打靶验证数据流: {TEST_DATA_PATH} ...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    df_test.columns = df_test.columns.str.strip()
    
    if 'MAX30100/1 raw' not in df_test.columns:
        print("❌ 错误：待验证测试 CSV 中必须包含 'MAX30100/1 raw' 列！")
        return
        
    ir_raw_test = df_test['MAX30100/1 raw'].values
    
    # 🧼 与训练完全一致的 EWM 高通滤波 + 一阶差分
    wave_hp_test = ir_raw_test - pd.Series(ir_raw_test).ewm(alpha=0.04).mean().values
    slope_test = np.diff(wave_hp_test, prepend=wave_hp_test[0])
    
    total_len = len(df_test)
    outputs_prob = np.zeros(total_len)
    
    print("🏎️  正在模拟单片机滚动滑窗前向计算 (无 PyTorch 框架干预)...")
    
    # 3. 🎬 滚动滑窗跑批
    for i in range(CNN_WINDOW_SIZE, total_len):
        w_win = wave_hp_test[i - CNN_WINDOW_SIZE : i]
        s_win = slope_test[i - CNN_WINDOW_SIZE : i]
        
        # 🧼 严格对齐的局域 Z-Score 能量配平
        w_norm = (w_win - np.mean(w_win)) / (np.std(w_win) + 1e-6)
        s_norm = (s_win - np.mean(s_win)) / (np.std(s_win) + 1e-6)
        
        # 拼装双通道特征：[2, 100]
        x_input = np.stack([w_norm, s_norm], axis=0)
        # 扩充为标准 Tensor 批次形状: [1, 2, 100]
        x_tensor = np.expand_dims(x_input, axis=0).astype(np.float32)
        
        # 💡 调用 ONNX 推理
        raw_outputs = session.run([output_name], {input_name: x_tensor})
        
        # 🎯 降维大招：使用 .item() 强行砸碎 ONNX 二维矩阵嵌套，释放被阉割的真实置信电压
        prob = raw_outputs[0].item()
        outputs_prob[i] = prob

    print("🏆 全盘时域计算结束！正在渲染多通道联动 Dashboard 图表...")

    # ==============================================================================
    # 🎨 4. 高精结果可视化大图表渲染
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 7))
    fig.suptitle("👨‍💻 Standard CPU ONNX Runtime Verification Dashboard", fontsize=14, fontweight='bold')
    
    # 上图：输入的滤波黄金生理波
    ax1.plot(wave_hp_test, color='#2ec4b6', label='MAX30100/2 Filtered PPG Signal', linewidth=1.2)
    ax1.set_ylim(-1000, 1000)
    ax1.set_ylabel('Amplitude', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')
    
    # 下图：对齐拉满后的 CNN 真实判定概率 (0.0 ~ 1.0)
    ax2.plot(outputs_prob, color='#ff1654', label='1D-CNN Peak Probability (0.0~1.0)', linewidth=1.5)
    ax2.axhline(y=0.82, color='gray', linestyle=':', label='MCU Trigger Threshold (0.82)')
    ax2.set_ylabel('CNN Confidence Score', fontweight='bold')
    ax2.set_xlabel('Timeline Samples Index (Time)', fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_onnx_cpu_verification()
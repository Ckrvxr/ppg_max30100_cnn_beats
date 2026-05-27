import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO


def run_labeling_tool(input_path, output_csv_path, view_window=1000):
    if input_path.endswith('.csv'):
        print(f"📂 正在载入已标记数据: {input_path} ...")
        df = pd.read_csv(input_path)
        df.columns = df.columns.str.strip()
        if 'IR' not in df.columns or 'RED' not in df.columns:
            raise ValueError("❌ CSV 必须包含 IR, RED 列")
        ir_raw = df['IR'].values.astype(np.float64)
        red_raw = df['RED'].values.astype(np.float64)
        total_len = len(df)
        beat_events = df['beat_event'].values.astype(int) if 'beat_event' in df.columns else np.zeros(total_len, dtype=int)
        print(f"📊 载入成功！总计 {total_len} 点, beat=1: {beat_events.sum()}")
    else:
        print(f"📂 正在载入原始数据: {input_path} ...")
        with open(input_path, 'rb') as f:
            raw = f.read()
        idx = raw.find(b'IR,RED')
        if idx == -1:
            raise ValueError("❌ 文件格式错误：找不到 'IR,RED' 标记")
        csv_part = raw[idx:].decode('ascii')
        df = pd.read_csv(StringIO(csv_part), header=None, names=['tag1', 'tag2', 'IR', 'RED'])
        ir_raw = df['IR'].values.astype(np.float64)
        red_raw = df['RED'].values.astype(np.float64)
        total_len = len(df)
        print(f"📊 载入成功！总计 {total_len} 点 ({total_len/6000:.1f} 分钟)")
        if os.path.exists(output_csv_path):
            df_exist = pd.read_csv(output_csv_path)
            if 'beat_event' in df_exist.columns and len(df_exist) == total_len:
                beat_events = df_exist['beat_event'].values.astype(int)
                print(f"📂 恢复已有标注: {beat_events.sum()} 个事件")
            else:
                beat_events = np.zeros(total_len, dtype=int)
        else:
            beat_events = np.zeros(total_len, dtype=int)

    wave_to_show = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values

    current_idx = 0
    print("\n=================== 👑 脉搏事件手动标注系统 ===================")
    print("  🖱️  左键单击：在波峰处标记脉搏事件")
    print("  🖱️  右键单击：擦除最近的标记")
    print("  ⌨️  ➔ 右方向键：保存并前进到下一页")
    print("  ⌨️  ⬅ 左方向键：回退上一页")
    print("  ⌨️  ESC：保存并退出")
    print("=============================================================\n")

    while current_idx < total_len:
        end_idx = min(current_idx + view_window, total_len)
        idx_range = np.arange(current_idx, end_idx)
        wave_slice = wave_to_show[current_idx:end_idx]

        fig, ax = plt.subplots(figsize=(15, 6))
        fig.suptitle(f"Labeling [Samples: {current_idx} ~ {end_idx} / Total: {total_len}]", fontsize=12, fontweight='bold')

        ax.plot(idx_range, wave_slice, color='#2ec4b6', linewidth=1.5, label='IR EWM Signal')
        ax.set_ylim(-2000, 2000)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Amplitude (IR EWM)", fontweight='bold')
        ax.set_xlabel("Timeline (Samples Index)", fontweight='bold')

        vlines = []
        current_beats = [idx for idx in idx_range if beat_events[idx] == 1]
        for beat in current_beats:
            vline = ax.axvline(x=beat, color='#ff1654', linestyle='-', linewidth=2, alpha=0.85)
            vlines.append((beat, vline))

        def onclick(event):
            if event.inaxes != ax: return
            x_click = int(round(event.xdata))
            if x_click < current_idx or x_click >= end_idx: return
            nonlocal vlines
            if event.button == 1:
                if beat_events[x_click] == 0:
                    beat_events[x_click] = 1
                    vline = ax.axvline(x=x_click, color='#ff1654', linestyle='-', linewidth=2, alpha=0.85)
                    vlines.append((x_click, vline))
                    fig.canvas.draw_idle()
                    print(f"📌 [标记] 样本点 {x_click}")
            elif event.button == 3:
                if len(vlines) == 0: return
                distances = [abs(v[0] - x_click) for v in vlines]
                min_idx = np.argmin(distances)
                if distances[min_idx] < 20:
                    target_beat, target_line = vlines[min_idx]
                    beat_events[target_beat] = 0
                    target_line.remove()
                    vlines.pop(min_idx)
                    fig.canvas.draw_idle()
                    print(f"🗑️  [撤销] 位置 {target_beat}")

        def onkey(event):
            nonlocal current_idx
            if event.key == 'right':
                current_idx += view_window
                plt.close(fig)
                out = pd.DataFrame({'IR': ir_raw, 'RED': red_raw, 'beat_event': beat_events})
                out.to_csv(output_csv_path, index=False)
            elif event.key == 'left':
                if current_idx >= view_window:
                    current_idx -= view_window
                    plt.close(fig)
            elif event.key == 'escape':
                current_idx = total_len + 1
                plt.close(fig)
                print("💾 正在保存当前进度...")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.tight_layout()
        plt.show()

        if current_idx > total_len:
            break

    out = pd.DataFrame({'IR': ir_raw, 'RED': red_raw, 'beat_event': beat_events})
    out.to_csv(output_csv_path, index=False)
    print(f"\n🏆 已标记数据集保存至: {output_csv_path} 🚀")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPG 脉搏事件手动标注工具')
    parser.add_argument('input', nargs='?', default='raw/raw_data.txt',
                        help='输入数据文件路径（.txt 或 .csv，默认 raw/raw_data.txt）')
    parser.add_argument('--output', '-o', default='labeled/labeled_data.csv',
                        help='输出标注 CSV 路径（默认 labeled/labeled_data.csv）')
    parser.add_argument('--view-window', type=int, default=1000,
                        help='每页显示采样点数（默认 1000）')
    args = parser.parse_args()
    run_labeling_tool(args.input, args.output, args.view_window)

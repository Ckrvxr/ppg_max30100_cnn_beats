import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 🎛️ 1000pts 黄金时轴大视图（100Hz下代表10秒波形，肉眼数峰最舒服的视觉比例）
VIEW_WINDOW = 1000  

def run_csv_manual_labeling(input_csv_path, output_csv_path):
    # 1. 直接用 pandas 读取你的未标注 CSV 文件
    print(f"📂 正在载入原始未标注数据: {input_csv_path} ...")
    df = pd.read_csv(input_csv_path)
    
    # 自动清洗表头可能存在的空格（防止读取不到列名）
    df.columns = df.columns.str.strip()
    
    if 'MAX30100/1 raw' not in df.columns:
        raise ValueError("❌ 错误：CSV 表头里必须包含 'MAX30100/1 raw' 列！")
        
    ir_raw = df['MAX30100/1 raw'].values
    total_len = len(df)
    print(f"📊 载入成功！总计包含数据点: {total_len} 个 (大约 {total_len/6000:.1f} 分钟)")
    
    # 🧼 离线高通平烫：把几万码的直流大漂移全部干掉，波形全程完美水平呼吸
    wave_to_show = ir_raw - pd.Series(ir_raw).ewm(alpha=0.04).mean().values
    
    # 2. 建立脉搏事件硬标签（支持增量存档，点累了关闭明天继续点，进度不会丢）
    if os.path.exists(output_csv_path):
        print("📂 检测到已存在的标注文件，正在恢复你之前的手工标注进度...")
        df_exist = pd.read_csv(output_csv_path)
        if 'beat_event' in df_exist.columns and len(df_exist) == total_len:
            beat_events = df_exist['beat_event'].values.copy()
        else:
            beat_events = np.zeros(total_len, dtype=int)
    else:
        # 🚫 初始全清空为 0，100% 纯净，没有任何机器算法干扰
        beat_events = np.zeros(total_len, dtype=int)

    current_idx = 0
    print("\n=================== 👑 脉搏事件 100% 纯手动标注系统 ===================")
    print("鼠标动作：")
    print("  🖱️  【鼠标左键单击】：在你认定的真脉搏波峰顶点点一下，【钉上一根红色事件线】。")
    print("  🖱️  【鼠标右键单击】：如果手抖点歪了，在红线附近点一下，【立刻将其撤销/擦除】。")
    print("键盘快捷键：")
    print("  ⌨️  【键盘 ➔ (右方向键)】：本页 1000pts 检查标记完毕，后台自动保存并滚动到下一页。")
    print("  ⌨️  【键盘 ⬅ (左方向键)】：回滚到上一页重新检查。")
    print("  ⌨️  【键盘 ESC】：提前退出并安全保存当前已打好的全部标签。")
    print("=======================================================================\n")

    while current_idx < total_len:
        end_idx = min(current_idx + VIEW_WINDOW, total_len)
        idx_range = np.arange(current_idx, end_idx)
        wave_slice = wave_to_show[current_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        fig.suptitle(f"Hand-Crafted Labeling Stream [Samples: {current_idx} ~ {end_idx} / Total: {total_len}]", fontsize=12, fontweight='bold')
        
        # 🟢 绘制纯净无杂质的绿色生理脉搏波
        ax.plot(idx_range, wave_slice, color='#2ec4b6', linewidth=1.5, label='PPG Live Wave (DC Cut)')
        low, high = np.percentile(wave_slice, [2, 98])
        ax.set_ylim(low, high)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Amplitude (DC Removed)", fontweight='bold')
        ax.set_xlabel("Timeline (Samples Index)", fontweight='bold')
        
        # 🔴 实时渲染你点出来的红色心跳垂直线
        vlines = []
        current_beats = [idx for idx in idx_range if beat_events[idx] == 1]
        for beat in current_beats:
            vline = ax.axvline(x=beat, color='#ff1654', linestyle='-', linewidth=2, alpha=0.85)
            vlines.append((beat, vline))
            
        # 鼠标左键点钉，右键拔钉逻辑
        def onclick(event):
            if event.inaxes != ax: return
            x_click = int(round(event.xdata))
            if x_click < current_idx or x_click >= end_idx: return
            
            nonlocal vlines
            if event.button == 1: # 左键：精准锁定波峰事件
                if beat_events[x_click] == 0:
                    beat_events[x_click] = 1
                    vline = ax.axvline(x=x_click, color='#ff1654', linestyle='-', linewidth=2, alpha=0.85)
                    vlines.append((x_click, vline))
                    fig.canvas.draw_idle()
                    print(f"📌 [标记] 样本点 {x_click} 被你认定为真脉搏主泵沿沿")
                    
            elif event.button == 3: # 右键：撤销/擦除
                if len(vlines) == 0: return
                distances = [abs(v[0] - x_click) for v in vlines]
                min_idx = np.argmin(distances)
                
                if distances[min_idx] < 20: # 容错范围内允许删除
                    target_beat, target_line = vlines[min_idx]
                    beat_events[target_beat] = 0
                    target_line.remove()
                    vlines.pop(min_idx)
                    fig.canvas.draw_idle()
                    print(f"🗑️  [撤销] 已成功擦除位置 {target_beat} 处的事件标记")

        # 键盘滚动与自动后台增量存档
        def onkey(event):
            nonlocal current_idx
            if event.key == 'right': # 前进一页
                current_idx += VIEW_WINDOW
                plt.close(fig)
                # 翻页时自动把当前进度重写回硬盘，绝对保障数据安全
                df['beat_event'] = beat_events
                df.to_csv(output_csv_path, index=False)
            elif event.key == 'left': # 后退一页
                if current_idx >= VIEW_WINDOW:
                    current_idx -= VIEW_WINDOW
                    plt.close(fig)
            elif event.key == 'escape': # 安全退出
                current_idx = total_len + 1
                plt.close(fig)
                print("💾 正在紧急安全导出已完成的部分...")

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        
        plt.tight_layout()
        plt.show()
        
        if current_idx > total_len:
            break
            
    # 终极重写保存
    df['beat_event'] = beat_events
    df.to_csv(output_csv_path, index=False)
    print(f"\n🏆【神级数据集制作完成】100%纯手动标记的数据集已安全保存至: {output_csv_path} 🚀")

# 🎬 填入你电脑里的 CSV 路径（确保里面的列名叫 ir_raw 和 red_raw）
run_csv_manual_labeling('2026-05-27_02-14-03.csv', 'cnn_perfect_dataset.csv')
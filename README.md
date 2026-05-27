# PPG 脉搏波检测工具

几个做 PPG 脉搏波检测的脚本，从数据标注到训练再到验证都包了。

模型是个轻量 CNN，跑完训练直接导出 `.pth` 和 C 头文件，方便塞进 MCU。

## 四个脚本

- **prelabel** — 拿 ONNX 模型先标一遍，省得从头点
- **label** — 打开图形界面手动标波峰，左键标、右键擦
- **train** — 把标好的数据扔进去训练，调参数啥的都在 `ppg_config.json` 里改
- **verify** — 看看模型推理效果，几种平滑方式对比着看

## 主要页面

![label](assets/label_1.avif)

![verify](assets/verify_1.avif)


## 用起来

```
pixi run train
pixi run verify
pixi run prelabel raw/data.txt
pixi run label
```

更详细的用法见 `docs/usage.md`。

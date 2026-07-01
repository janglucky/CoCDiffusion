<div align=center class="logo">
      <img src="figs/logo1.png" style="width:640px">
</div>

## SeeSR Deblur Variant

这个仓库基于 [cswry/SeeSR](https://github.com/cswry/SeeSR) 改造，当前版本只保留图像去模糊任务：
- 不使用文本分支
- 不保留 `null_text`
- 不做图像上采样
- 训练使用原始图像尺寸
- 测试使用原始图像尺寸
- 数据集直接支持 `source/target` 成对目录

当前仓库已经针对本地环境和数据路径补好了可运行脚本，并完成了训练、测试、评估联通性验证。

## 环境
```bash
git clone https://github.com/cswry/SeeSR.git
cd SeeSR

conda create -n seesr python=3.10
conda activate seesr
pip install -r requirements.txt
```

说明：仓库内不再 vendored `basicsr/` 源码目录；PSNR / SSIM 评估使用 pip 安装的 `basicsr==1.4.2`。

## 模型路径
当前脚本默认使用以下本地模型路径：
- Stable Diffusion 2 Base: `/home/gd09385/models/stable-diffusion-2-base`
- SeeSR 权重: `/home/gd09385/models/seesr`

如果你要改路径，可以通过环境变量覆盖 `scripts/*.sh` 里的默认值。

## 数据组织
训练和评估数据默认使用：`/home/gd09385/data/train_c_sub`

目录结构：
```text
/home/gd09385/data/train_c_sub/
├── source/
│   ├── xxx.png   # 模糊图
│   └── ...
└── target/
    ├── xxx.png   # 清晰图
    └── ...
```

要求：
- `source` 和 `target` 中同名文件一一对应
- 每对图像尺寸一致
- 支持 `png/jpg/jpeg/bmp/webp`

## 测试
对整个 `source` 目录做去模糊：
```bash
bash scripts/test_seesr.sh
```

对单张图片做去模糊：
```bash
IMAGE_PATH=/home/gd09385/data/train_c_sub/source/1P0A0890_s003.png \
OUTPUT_DIR=/home/gd09385/work/CoCDiffusion/experience/my_test \
bash scripts/test_seesr.sh
```

说明：
- 推理默认保持原始分辨率
- 如果宽高不是 8 的倍数，只会做最小边缘 padding，输出后裁回原尺寸
- 不再有 `upscale` 或 `process_size` 逻辑

## 训练
直接使用 `source/target` 做训练：
```bash
bash scripts/train_seesr.sh
```

常用覆盖方式：
```bash
OUTPUT_DIR=/home/gd09385/work/CoCDiffusion/experience/deblur_run1 \
TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_TRAIN_STEPS=10000 \
bash scripts/train_seesr.sh
```

脚本默认：
- 训练默认优化 ControlNet
- 已禁用文本分支相关计算
- 默认数据根目录为 `/home/gd09385/data/train_c_sub`

## 评估
对预测结果和 `target` 计算 PSNR / SSIM：
```bash
bash scripts/eval_seesr.sh
```

指标实现来自 pip 包 `basicsr.metrics`，本地仓库只保留轻量评估入口。

如果测试结果在别的目录：
```bash
PREDICTION_PATH=/home/gd09385/work/CoCDiffusion/experience/deblur_test_c_sub/sample00 \
TARGET_PATH=/home/gd09385/data/train_c_sub/target \
bash scripts/eval_seesr.sh
```

也可以直接调用 Python：
```bash
python eval_seesr.py \
  --prediction_path /home/gd09385/work/CoCDiffusion/experience/deblur_test_c_sub/sample00 \
  --target_path /home/gd09385/data/train_c_sub/target \
  --verbose
```

## 已验证的联通性
以下验证已在 `seesr` 虚拟环境完成：
- 训练 smoke test 成功，保存 checkpoint
- 测试 smoke test 成功，输出原尺寸结果
- `scripts/train_seesr.sh` 成功
- `scripts/test_seesr.sh` 成功
- `scripts/eval_seesr.sh` 成功

对应输出目录：
- 训练 smoke test: `/home/gd09385/work/CoCDiffusion/experience/smoke_train_c_sub_original_size`
- 测试 smoke test: `/home/gd09385/work/CoCDiffusion/experience/smoke_test_c_sub_original_size`
- 脚本训练 smoke test: `/home/gd09385/work/CoCDiffusion/experience/script_smoke_train_c_sub`
- 脚本测试 smoke test: `/home/gd09385/work/CoCDiffusion/experience/script_smoke_test_c_sub`

## 关键改动
- 删除文本提示输入链路，不再依赖标签文本
- 删除 `null_text` 方案，避免文本分支 FLOPs
- 删除 RAM/DAPE 图像语义分支，ControlNet 直接使用原始模糊图作为条件输入
- 测试改为原尺寸去模糊
- 数据加载器支持 `source/target` 并按文件名配对

## 致谢
本项目改造自 [SeeSR](https://github.com/cswry/SeeSR)，原始工作基于 diffusers、BasicSR、PASD、RAM 等项目，在此一并致谢。

<div align=center class="logo">
      <img src="figs/logo1.png" style="width:640px">
</div>

## CoCDiffusion

本仓库基于 [cswry/SeeSR](https://github.com/cswry/SeeSR) 改造，当前目标只保留单图像去模糊任务。默认实验已经切换为 `coc_image_latent`：在图像域使用 CoC 离焦模型生成逐步模糊图，再编码成 latent，作为官方 diffusers scheduler 中的替代噪声。

当前版本的主要特点：
- 删除文本分支和 `null_text`，避免文本侧 FLOPs
- 删除 RAM/DAPE 图像语义分支，ControlNet 直接以原始模糊图为条件
- 不做图像上采样，训练和测试都使用原始图像尺寸
- 支持原始 DDIM/DDPM、CoC blur、paired endpoint、CoC endpoint、CoC image-latent 等实验路径
- 默认训练目标只保留 latent-space MSE，不再使用图像重建损失或 SSIM 损失

## 方法概览

默认 `coc_image_latent` 前向过程为：

```text
z_0 = VAE(gt)
epsilon_coc,t = VAE(CoCBlur(gt, depth, t))
epsilon_coc,t = Normalize(epsilon_coc,t)
x_t = scheduler.add_noise(z_0, epsilon_coc,t, t)
```

其中：
- `gt` 是清晰目标图像
- `depth` 是对应深度图
- `CoCBlur(gt, depth, t)` 在图像域执行离焦模糊
- `x_t` 是 latent-space 扩散状态
- `scheduler.add_noise` 使用 diffusers 官方 scheduler 的 `alpha_prod_t`
- CoC 模糊强度仍由 `blur_scale_t = (t / (T - 1)) ^ schedule_power` 控制
- `Normalize` 默认对每个样本的 latent 做零均值、单位方差标准化，使 CoC 噪声 latent 的尺度更接近 `N(0, I)`

网络输出对齐标准 DDIM/DDPM 的噪声估计语义：在 `coc_image_latent` 中不预测 clean latent，而是预测当前 timestep 的 CoC 模糊 latent `epsilon_coc,t`。反向过程直接沿用官方 scheduler：

```text
epsilon_hat = model(x_t, t, source)
x_{t-1} = scheduler.step(epsilon_hat, t, x_t)
```

这样 CoC 只负责生成图像域模糊，扩散混合和反向采样全部交给 diffusers 官方 scheduler。

注意：标准化只能匹配均值和方差，不能严格保证 CoC latent 成为独立高斯噪声；它是为了让结构化 CoC 噪声与 DDIM/DDPM scheduler 的数值尺度更兼容。

也可以测试更贴近 CoC 前向轨迹的逆向过程，但它不是默认路径：

```text
epsilon_hat_t = model(x_t, t, source)
z_0_hat = (x_t - sqrt(1 - alpha_t) * epsilon_hat_t) / sqrt(alpha_t)
epsilon_hat_{t-1} = Normalize(VAE(CoCBlur(Decode(z_0_hat), depth, t-1)))
x_{t-1} = sqrt(alpha_{t-1}) * z_0_hat + sqrt(1 - alpha_{t-1}) * epsilon_hat_{t-1}
```

该模式通过 `--coc_image_latent_reverse recompute_prev` 开启，需要测试时提供 depth；否则会自动退回官方 scheduler 逆向。当前 checkpoint 上该路径容易因为每步 `Decode -> CoCBlur -> Encode` 的闭环误差而塌缩，建议只作为诊断实验。

新增一个更轻量的端点锚定逆向：

```text
epsilon_hat_t = model(x_t, t, source)
z_0_hat = (x_t - sqrt(1 - alpha_t) * epsilon_hat_t) / sqrt(alpha_t)
x_{t-1} = sqrt(alpha_{t-1}) * z_0_hat + sqrt(1 - alpha_{t-1}) * VAE(source)
```

该模式通过 `--coc_image_latent_reverse source_endpoint` 开启，不需要 depth。它把真实输入作为退化端点，比 `recompute_prev` 稳定，但高 timestep 仍会放大 `z_0_hat` 误差；可以配合 `START_STEPS` 从较低退化程度开始。

## 环境

```bash
conda create -n seesr python=3.10
conda activate seesr
pip install -r requirements.txt
```

本地脚本默认会自动激活：

```bash
source /home/gd09385/anaconda3/bin/activate seesr
```

说明：仓库内不再 vendored `basicsr/` 源码目录；PSNR/SSIM 评估使用 pip 安装的 `basicsr==1.4.2`。

## 模型路径

脚本默认使用以下本地模型路径：
- Stable Diffusion 2 Base: `/home/gd09385/models/stable-diffusion-2-base`
- SeeSR 初始权重: `/home/gd09385/models/seesr`

可以通过环境变量覆盖脚本默认值，例如：

```bash
PRETRAINED_MODEL_PATH=/path/to/stable-diffusion-2-base \
SEESR_MODEL_PATH=/path/to/seesr \
bash scripts/train_coc.sh
```

## 数据组织

默认训练数据路径：`/home/gd09385/data/test_c_sub`

`coc_image_latent`、`coc_endpoint`、`coc_blur` 训练使用三元组数据：

```text
/path/to/dataset/
├── source/
│   ├── xxx.png   # 模糊输入图
│   └── ...
├── target/
│   ├── xxx.png   # 清晰目标图
│   └── ...
└── depth/
    ├── xxx.png   # 深度图
    └── ...
```

要求：
- `source`、`target`、`depth` 中同名文件一一对应
- `source` 和 `target` 尺寸一致
- `depth` 尺寸与图像一致
- 支持 `png/jpg/jpeg/bmp/webp`

原始 DDIM 或 `paired_endpoint` 可以使用成对数据：

```text
/path/to/dataset/
├── source/
└── target/
```

## 训练

默认训练 `coc_image_latent`：

```bash
bash scripts/train_coc.sh
```

等价入口：

```bash
bash scripts/train_seesr.sh
```

常用覆盖方式：

```bash
ROOT_FOLDERS=/home/gd09385/data/test_c_sub \
OUTPUT_DIR=/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_image_latent \
TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
MAX_TRAIN_STEPS=10000 \
bash scripts/train_coc.sh
```

脚本默认值：
- `DIFFUSION_PROCESS=coc_image_latent`
- `UNET_TRAIN_PRESET=controlnet_interaction_full`
- `CHECKPOINTING_STEPS=5000`
- `TIMESTEP_CONDITIONING=auto`，对 `coc_image_latent` 默认开启 timestep embedding
- `COC_NOISE_NORMALIZATION=sample`，对 CoC blur latent 做每样本零均值/单位方差标准化

当前训练损失只包含 latent MSE；`coc_image_latent` 的目标是图像域 CoC blur 后再编码并标准化得到的 `epsilon_coc,t`：

```text
loss = MSE(model_pred, epsilon_coc,t)
```

不再计算图像重建损失和 SSIM 损失。

## 测试

默认测试入口：

```bash
bash scripts/test_coc.sh
```

等价入口：

```bash
bash scripts/test_seesr.sh
```

脚本默认：
- 模型路径: `/home/gd09385/work/CoCDiffusion/experiment/deblur_train_coc_image_latent/checkpoint-5000`
- 输入路径: `/home/gd09385/data/test_c/source`
- 输出路径: `/home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_image_latent-lr-5000-1`
- `DIFFUSION_PROCESS=coc_image_latent`
- `NUM_INFERENCE_STEPS=1`
- `COC_IMAGE_LATENT_REVERSE=scheduler`
- `COC_IMAGE_LATENT_NORMALIZE_START=0`

`coc_blur` / `coc_endpoint` 旧实验路径使用 depth 测试：

```bash
USE_DEPTH=1 \
DEPTH_PATH=/home/gd09385/data/test_c/depth \
bash scripts/test_coc.sh
```

`coc_image_latent` 默认测试时不需要 depth；输入起点是 raw `VAE(source)`，反向采样使用 diffusers 官方 scheduler。

如果要测试显式重建 `t-1` CoC latent 的逆向过程：

```bash
USE_DEPTH=1 \
DEPTH_PATH=/home/gd09385/data/test_c/depth \
bash scripts/test_coc.sh --coc_image_latent_reverse recompute_prev
```

对照官方 scheduler 逆向：

```bash
bash scripts/test_coc.sh --coc_image_latent_reverse scheduler
```

测试端点锚定逆向，并从中等退化程度开始：

```bash
NUM_INFERENCE_STEPS=20 \
START_STEPS=201 \
COC_IMAGE_LATENT_REVERSE=source_endpoint \
bash scripts/test_coc.sh
```

如果要强制一步从最大训练 timestep 开始，可使用 full-range 时间表：

```bash
NUM_INFERENCE_STEPS=1 \
COC_IMAGE_LATENT_REVERSE=source_endpoint \
COC_IMAGE_LATENT_TIMESTEP_SPACING=full_range \
bash scripts/test_coc.sh
```

多步测试示例：

```bash
NUM_INFERENCE_STEPS=20 \
OUTPUT_DIR=/home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_image_latent-5000-step20 \
bash scripts/test_coc.sh
```

推理保持原始分辨率。如果宽高不是 8 的倍数，只做最小边缘 padding，输出后裁回原尺寸。

## DDIM Baseline

如果只看原始 Gaussian DDIM/DDPM baseline，使用 `gaussian` 分支即可。训练前向过程为：

```text
z_0 = VAE(gt)
epsilon ~ N(0, I)
x_t = scheduler.add_noise(z_0, epsilon, t)
loss = MSE(model(x_t, t, source), epsilon)
```

测试反向过程同步使用官方 scheduler：

```text
epsilon_hat = model(x_t, t, source)
x_{t-1} = scheduler.step(epsilon_hat, t, x_t)
```

训练 baseline：

```bash
bash scripts/train_ddim.sh
```

默认训练设置：
- `DIFFUSION_PROCESS=gaussian`
- `TIMESTEP_CONDITIONING=on`
- `ROOT_FOLDERS=/home/gd09385/data/train_c_sub`
- 默认只训练 ControlNet；如需解冻 UNet，可设置 `UNET_TRAIN_PRESET`

测试已有 baseline checkpoint：

```bash
bash scripts/test_ddim.sh
```

当前本地已有 checkpoint：
- `/home/gd09385/work/CoCDiffusion/experiment/deblur_train_c_sub/checkpoint-36000`

单图连通性测试：

```bash
IMAGE_PATH=/home/gd09385/data/test_c/source/1P0A0917.png \
OUTPUT_DIR=/home/gd09385/work/CoCDiffusion/experiment/debug_ddim_baseline_one \
NUM_INFERENCE_STEPS=20 \
bash scripts/test_ddim.sh
```

说明：`/home/gd09385/data/test_c` 的 `source` 和 `target` 文件名不完全同名，不能直接用 `eval_seesr.py` 自动配对。按排序粗配第一张图时，输入约为 `25.25 dB / 0.662 SSIM`，baseline 输出约为 `23.00 dB / 0.557 SSIM`，主要用于判断趋势而非最终论文指标。

## 其他实验路径

可以通过 `DIFFUSION_PROCESS` 切换：

```bash
DIFFUSION_PROCESS=gaussian bash scripts/train_coc.sh
DIFFUSION_PROCESS=coc_blur bash scripts/train_coc.sh
DIFFUSION_PROCESS=paired_endpoint bash scripts/train_coc.sh
DIFFUSION_PROCESS=coc_endpoint bash scripts/train_coc.sh
DIFFUSION_PROCESS=coc_image_latent bash scripts/train_coc.sh
```

说明：
- `gaussian` 使用原始 DDIM/DDPM 加噪路径
- `coc_blur` 使用 CoC blur cold diffusion
- `paired_endpoint` 在 clean/source latent 之间做 endpoint cold diffusion
- `coc_endpoint` 使用 CoC 轨迹并以真实 source latent 作为端点
- `coc_image_latent` 在图像域 CoC blur，再编码成 latent 替换官方 DDIM/DDPM 中的噪声，是当前默认方法

## CoC 前向可视化

可视化 CoC 加模糊过程：

```bash
bash scripts/visualize_coc_forward.sh
```

默认输出：

```text
/home/gd09385/work/CoCDiffusion/experiment/coc_forward_visualization
```

该脚本可用于检查模糊半径随时间增长、景深变化、全局模糊覆盖等行为。

## 评估

对预测结果和 `target` 计算 PSNR/SSIM：

```bash
bash scripts/eval_seesr.sh
```

如果测试结果在别的目录：

```bash
PREDICTION_PATH=/home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_image_latent-5000-onestep/sample00 \
TARGET_PATH=/home/gd09385/data/test_c/target \
bash scripts/eval_seesr.sh
```

也可以直接调用 Python：

```bash
python eval_seesr.py \
  --prediction_path /home/gd09385/work/CoCDiffusion/experiment/deblur_test_coc_image_latent-5000-onestep/sample00 \
  --target_path /home/gd09385/data/test_c/target \
  --verbose
```

指标实现来自 pip 包 `basicsr.metrics`。

## 关键文件

- `train_seesr.py`: 训练入口，支持多种 diffusion process
- `test_seesr.py`: 测试入口，支持可选 depth
- `pipelines/pipeline_seesr.py`: 推理 pipeline 和反向过程
- `coc.py`: 图像域 CoC 离焦渲染器和 `add_coc_blur` 函数
- `schedulers/coc_endpoint_scheduler.py`: CoC endpoint 对照 scheduler
- `schedulers/paired_endpoint_scheduler.py`: paired endpoint 对照 scheduler
- `dataloaders/triplet_dataset.py`: `source/target/depth` 三元组数据集
- `dataloaders/paired_dataset.py`: `source/target` 成对数据集

## 已验证检查

当前版本已完成以下轻量检查：
- `python -m py_compile train_seesr.py test_seesr.py pipelines/pipeline_seesr.py coc.py`
- `bash -n scripts/train_coc.sh && bash -n scripts/test_coc.sh`
- `python test_seesr.py --help` 中已包含 `coc_image_latent`
- `coc_image_latent` 使用官方 scheduler sanity check：CoC blur latent 作为 `epsilon` 传入 `add_noise/step`

## 致谢

本项目改造自 [SeeSR](https://github.com/cswry/SeeSR)，原始工作基于 diffusers、BasicSR、PASD、RAM 等项目，在此一并致谢。

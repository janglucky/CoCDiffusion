Default inference settings
```
python test_seesr.py \
--pretrained_model_path /home/gd09385/models/stable-diffusion-2-base \
--seesr_model_path /home/gd09385/models/seesr \
--image_path /home/gd09385/data/train_c_sub/source \
--output_dir /home/gd09385/work/CoCDiffusion/experience/deblur_test_c_sub \
--start_point lr \
--num_inference_steps 50
```

当前仓库已经改为去模糊任务：
- 不使用文本分支
- 不使用 `null_text`
- 不使用 RAM/DAPE 图像语义分支
- 不做图像上采样
- 输入保持原始尺寸
- 如果尺寸不是 8 的倍数，只会做最小 padding，并在输出后裁回原尺寸

常用参数
- `--num_inference_steps`
  - 采样步数越大，生成结果通常更强，但耗时也更高。
  - 当前默认值是 `50`。
  - 如果想做快速联通性测试，可以设为 `2`。
- `--conditioning_scale`
  - 控制条件分支强度，默认 `1.0`。
- `--align_method`
  - 可选 `wavelet`、`adain`、`nofix`。
  - 默认 `adain`。
- `--sample_times`
  - 每张图生成多少次，默认 `1`。

当前版本已经删除的旧参数逻辑
- `--upscale`
- `--process_size`

训练说明
```
accelerate launch train_seesr.py \
--pretrained_model_name_or_path /home/gd09385/models/stable-diffusion-2-base \
--controlnet_model_name_or_path /home/gd09385/models/seesr \
--unet_model_name_or_path /home/gd09385/models/seesr \
--output_dir /home/gd09385/work/CoCDiffusion/experience/deblur_train_c_sub \
--root_folders /home/gd09385/data/train_c_sub \
--enable_xformers_memory_efficient_attention \
--mixed_precision fp16 \
--learning_rate 1e-5 \
--train_batch_size 1 \
--gradient_accumulation_steps 1
```

训练数据目录要求
```text
train_c_sub/
├── source/
└── target/
```
- `source` 是模糊图
- `target` 是清晰图
- 同名文件自动按文件名配对

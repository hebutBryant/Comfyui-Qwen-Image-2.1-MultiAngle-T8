# Qwen-Image 2.1 多参考融合与双人互动实验记录

整理日期：2026-10-11。本文汇总本次对话中的用户指令、实际执行方法、参数、参考图片与结果路径，并记录与 `02-comfy-pipeline` 实现的区别。实验结果依据已保存的脚本、清单、运行记录与验证文件；本次整理没有重新运行推理。

## 1. 用户指令与实验目标

以下按对话顺序摘录用户指令，保留原意：

1. “详细解释一下这个项目”。
2. “他是如何实现多个素材融合到一张image中的”。
3. “目前不用AnyAngle LoRA，只用Qwen-Image2.1: /home/hdd4/lpz/Qwen-Image-2.1把多个参考图元素放在一个image和scene中，参考元素和背景需要你自己生成决定，请你用这个项目中的方法跑一下”。
4. “继续”；“现在显卡有空闲”。
5. “四张独立参考分别是湖畔木桌背景、青绿色陶瓷壶、橙色帆布包、红花玻璃瓶 告诉我一下这几个参考图片的绝对路径”。
6. “如果再现有的场景下再生成两个人”；“如果再现有的场景下再生成两个人,他们直接发生交互关系，是否能够生成好”。
7. “生成一个人指着桌子上的花，一个人观看 生成一下”。
8. “/home/lipz/DraftDAG/DraftDAG/explorations/02-comfy-pipeline 对比这个文件夹的实现方法，两种多图融合的实现方法有什么不同”。
9. “把我做的实验和指令形成一个md文件，写到这个路径中”。

核心要求：仅使用本地 Qwen-Image 2.1，生成独立参考素材并融合为完整场景，不加载 AnyAngle LoRA；随后在已生成场景中加入两个人，使一人指花、另一人观看。

## 2. 环境与执行边界

| 项目 | 实际设置 |
| --- | --- |
| 当前项目 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8` |
| 模型权重 | `/home/hdd4/lpz/Qwen-Image-2.1` |
| Python | `/mnt/nfs0/wangmh2/envs/flux2_env/bin/python` |
| GPU | NVIDIA RTX A6000，48 GB |
| 框架 | PyTorch `2.14.0+cu130`、Diffusers `0.41.0.dev0`、Transformers `5.17.0` |
| 精度与内存设置 | bf16、模型 CPU offload、VAE tiling、前缀 KV cache |
| LoRA | 两次实验均未加载 |
| 模型加载 | 本地文件模式，未调用在线图像服务 |

两次实验复用了项目的 `canonical_scene()`、`build_manifest()`、`prompt_for()` 和 `StudioStore`，由项目生成素材清单、用途指令和便携场景 ZIP。实际推理由 Diffusers `QwenImage21Pipeline` 执行：独立图片列表通过 `image=[...]` 输入。

本次没有启动 ComfyUI 服务，也没有直接执行 `AnyAngleMultiPersonEncodeT8` 节点。因此，这是复用项目参考组织方法的本地 Diffusers 实跑，不能声称与 ComfyUI 节点逐像素一致。

## 3. 实验一：四张独立参考融合成湖畔静物场景

### 3.1 素材与绝对路径

四张参考均由本地 Qwen 2.1 分别生成；输入模型前没有拼成一张图。

| 图序 | 内容 | 原始尺寸 | 绝对路径 |
| --- | --- | --- | --- |
| image1 | 湖畔木桌背景 | 768×512 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/references/background.png` |
| image2 | 青绿色陶瓷壶 | 640×640 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/references/teapot.png` |
| image3 | 橙色帆布包 | 640×640 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/references/bag.png` |
| image4 | 红花玻璃瓶 | 640×640 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/references/flowers.png` |

### 3.2 融合方法与指令

场景采用 `references-only` 模式，`conditioning.model = base`，不使用三维引导图、区域遮罩或 ControlNet。背景绑定 scene 用途，三件物品分别绑定 accessory 用途，并通过文字目标指定桌面左、中、右的位置。

提示词要求：保留湖泊、山景、木桌与暖色光照；桌上恰好放置三件物品，从左到右依次为青绿色壶、橙色包、红花玻璃瓶；统一透视、尺度、光照和接触阴影，保留各自颜色、形状与材质，包为最大物件。物品参考中的摄影背景不应带入成图，不生成多面板、重复物体或人物。

图片按照项目面积预算公式缩放，首图和其他图的预算均为 640。这里的 640 是面积预算参数，不表示全部参考被强制缩放为 640×640。每张参考保持独立，参与视觉语言编码与参考 VAE 编码；位置和用途通过提示词表达，不构成逐素材的硬区域约束。

完整原始提示词与素材生成种子保存在实验目录的 `fusion_prompt.txt`、`reference_prompts.json`，具体编码输入尺寸保存在 `encoding_inputs.json`。

### 3.3 参数与结果

| 设置 | 素材生成 | CFG 1 融合 | CFG 3 融合 |
| --- | --- | --- | --- |
| 步数 | 每张 24 | 32 | 32 |
| CFG | 1 | 1 | 3 |
| 融合 seed | 不适用，各图单独记录 | 2026101142 | 2026101142 |
| 最终尺寸 | 背景 768×512，物品 640×640 | 768×512 | 768×512 |
| GPU | 物理 GPU 1 | 物理 GPU 1 | 物理 GPU 1 |
| 融合耗时 | 不计入右侧耗时 | 约 52.2 秒 | 约 60.5 秒 |

两版成图均呈现一个完整场景，壶在左、包居中、花瓶在右，湖景和木桌保留，三件物品完整可见。壶的竹柄、包的浅色前袋、透明瓶和红花保留主要特征。模型会重绘比例、纹理、把手和光影，不能保证参考像素原样复制；部分区域仍可观察到轻微色带。

最初 CFG 3 运行因其他任务临时占用 GPU 1 约 34 GB 而发生 OOM，未产生成图。随后 CFG 1 成功；显卡空闲后 CFG 3 重跑成功，未停止其他 GPU 任务。

结果绝对路径：

- CFG 1：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/final-scene-seed-2026101142-cfg-1.png`
- CFG 3：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/final-scene-seed-2026101142-cfg-3.png`
- 展示对照：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo/comparison.jpg`

验证记录确认场景 ZIP 可导入、四张参考可回读且编号顺序正确、PNG 提示词与场景一致、输出 SHA256 一致。此验证检查工程产物有效性，图像语义质量另由目视检查判断。

## 4. 实验二：现有场景加入两人，男子指花、女子观看

### 4.1 输入参考

使用实验一 CFG 3 成图作为场景参考，另由本地 Qwen 2.1 生成两张虚构人物参考。男性穿深蓝外衫和白色 T 恤，女性穿芥末黄色开衫和米白上衣。

| 图序 | 内容 | 绝对路径 |
| --- | --- | --- |
| image1 | 已生成的湖畔静物场景副本 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/references/scene.png` |
| image2 | 指花男性参考 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/references/man.png` |
| image3 | 看花女性参考 | `/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/references/woman.png` |

场景图绑定 scene 用途；每张人物图分别绑定 identity 与 clothing 两种用途。同一张人物图只作为一个图像输入发送，提示词分别说明身份与服装用途。

### 4.2 互动指令

提示词具体要求：

- 桌后恰好出现两个成年人，男子位于桌后左半边，女子位于右半边，下半身由桌子遮挡。
- 男子转头看花，伸出右臂，右手食指朝一朵红花指去，其他手指自然弯曲，指尖停在花前，不接触瓶身或花瓣。
- 女子顺着指向，转头并低头看同一簇红花，带轻微感兴趣的微笑；两人看花而不是看镜头。
- 指花的手和花朵应清楚可见，前臂不遮住脸。
- 保留湖景、木桌和原有壶、包、花瓶，统一透视与自然暖光，不新增人物、手臂、物体或面板。

完整原始提示词：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/interaction_prompt.txt`。

这次互动由文字关系和三个独立参考共同条件化生成，没有 SAM 抠图、人物贴图、三维姿势引导或区域遮罩。

### 4.3 参数与结果

人物参考各 640×640，24 步、CFG 1；最终生成使用物理 GPU 0，40 步、CFG 3，seed 为 2026101180，输出 768×512。互动成图耗时约 54.9 秒，不包含模型加载与人物参考生成。

目视检查：成图中正好两人，男子食指朝红花伸出，女子低头看花；原有壶、包、花瓶均保留，顺序仍为左、中、右。相对输入场景，物件尺寸、位置和桌面构图有所调整。尽管提示词要求保持原位置，本次方法仍不能严格锁定背景或物件像素。

该结果支持“这次样例能够生成所要求的互动”，不能由单次结果推断不同 seed、更多人物或更复杂动作都同样稳定。

结果绝对路径：

- 成图：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/interaction-seed-2026101180-cfg-3.png`
- 展示对照：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/comparison-seed-2026101180.jpg`
- 参数与哈希：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction/result-seed-2026101180-cfg-3.json`

## 5. 与 02-comfy-pipeline 的实现对比

对比目录：`/home/lipz/DraftDAG/DraftDAG/explorations/02-comfy-pipeline`。

两种实现的主要区别在于模型收到的参考组织方式：本次实验将多个素材作为独立参考交给模型，并用文字分配用途与关系；对比目录先抠取人物、确定各层位置并合成一张画布，再让模型融合这张已有布局的图。

```text
本次实验：
背景参考 + 独立物品/人物参考
    → 项目素材清单与用途/关系提示词
    → Qwen 2.1 多参考输入
    → 完整场景

02-comfy-pipeline：
背景 + 五张人物参考
    → SAM2.1 分割获得人物 alpha
    → 按 Fabric transforms 缩放、定位、叠层
    → 一张人物已排布的拼贴画布
    → 图像条件编码与模型融合
    → 完整场景
```

| 对比项 | 本次独立参考实验 | 02-comfy-pipeline |
| --- | --- | --- |
| 输入组织 | 四张静物参考，或场景加两张人物参考，独立送入模型 | 多个素材在模型融合前合成一个场景画布 |
| 前景分割 | 未使用 | 使用本地 SAM2.1，生成 RGBA 人物素材 |
| 初始布局 | 左中右位置、人物站位通过文字表达 | Fabric transforms 明确给出位置、缩放、层序等 |
| 素材用途 | 独立图编号对应 scene、accessory、identity、clothing 等用途 | 人物和背景已在画布上形成像素空间关系 |
| 动作关系 | 由提示词要求模型重新生成手势、朝向与视线 | 初始姿势来自人物素材，合成器只改变摆放，后续模型可重绘 |
| 本次实际后端 | Diffusers QwenImage21Pipeline | 原生主流程调用 ComfyUI 编码、采样与解码节点；另有 Diffusers Qwen 2.1 对照 |
| 本次是否使用三维引导 | 未使用 | 此目录采用二维离线图层合成 |
| 遮罩含义 | 无区域遮罩 | SAM alpha 用于抠图与像素合成，不能因此认定为扩散区域约束 |
| 背景和位置保持 | 存在重绘和位置漂移 | 初始拼贴位置明确，但最终模型输出仍不保证像素锁定 |
| LoRA | 无 | 主流程已删除 Lightning 与 Put-it-here LoRA 节点 |

### 5.1 对比目录有两条不同的模型路径

1. **原生工作流主结果**：素材由本地 Qwen 2.1 生成，最终融合使用 Qwen-Image-Edit-2511 GGUF Q4_K_M，搭配原工作流文本编码器与 VAE。主结果选用 8 步、CFG 1，尺寸为 2000×920；40 步、CFG 4 对照出现灰暗、颗粒等伪影。两版文件分别为 `runs/001-autumn-lounge/final_no_lora_8step.png`、`final_no_lora_40step.png`，选中的副本为 `final_scene.png`。
2. **Qwen 2.1 对照结果**：`src/generate_assets.py` 的 `--edit` 分支读取一张拼贴图，作为单个 PIL 图像传给同一个本地 Qwen 2.1 模型。记录结果为 `runs/001-autumn-lounge/final_qwen21.png`，1536×704、30 步、CFG 1，seed 为 202610118。

因此，比较参考融合方法时，可以对照“独立参考输入 Qwen 2.1”与“拼贴单图输入 Qwen 2.1”。如果拿本次结果与该目录原生主结果比较，还同时改变了模型、量化、编码器、采样参数、素材、分辨率与场景，不能把质量差异全部归因于参考组织方法。

### 5.2 编码与合成细节

`src/image_nodes.py` 的 `SAM2BackgroundRemoval` 使用躯干正点、两侧负点和框提示分割人物；对白色摄影背景的先验只用于候选遮罩排序，最终 alpha 来自选中的 SAM 遮罩。

同文件的 `composite()` 读取保存的 Fabric transforms，使用 Pillow 进行缩放、定位与 alpha 叠层。当前离线实现覆盖该实验用到的变换，非零旋转和倾斜等未实现设置会报错，不能概括为完整浏览器合成器的所有功能。

原生主流程通过 `TextEncodeQwenImageEditPlus` 编码合成图，并由 `ReferenceLatent` 追加全尺寸 VAE latent。两份 latent 是同一合成场景的不同参考表示，不等于五个人物各自保留一个独立参考槽。

对于“一个人指花，另一个人看花”的任务，独立参考方式让人物身份与服装保持独立，动作由模型按关系词重新生成。二维拼贴方式能先明确人物位置与尺度，但仅移动静态图层不会产生新的指花手势；仍需要合适的源姿势或后续模型重绘。此处是依据两种处理方式作出的分析，未在对比目录追加双人互动实验。

## 6. 本次实验复跑命令

从项目根目录执行，已有参考会复用；需选择可用显卡。

```bash
cd /home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8

# 四图融合：CFG 3，脚本默认物理 GPU 1
/mnt/nfs0/wangmh2/envs/flux2_env/bin/python -u -B \
  .local/qwen21-multireference-demo/run_demo.py \
  --fusion-steps 32 --fusion-seed 2026101142 --fusion-cfg 3

# 四图融合：CFG 1 对照
/mnt/nfs0/wangmh2/envs/flux2_env/bin/python -u -B \
  .local/qwen21-multireference-demo/run_demo.py \
  --fusion-steps 32 --fusion-seed 2026101142 --fusion-cfg 1

# 两人互动：物理 GPU 0
/mnt/nfs0/wangmh2/envs/flux2_env/bin/python -u -B \
  .local/qwen21-two-person-interaction/run_interaction.py \
  --gpu 0 --steps 40 --cfg 3 --seed 2026101180
```

## 7. 原始记录索引

静物实验目录：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-multireference-demo`。

- `README.md`、`run_demo.py`：执行说明与脚本。
- `reference_prompts.json`、`fusion_prompt.txt`：素材与融合的完整原始提示词。
- `scene.json`、`manifest.json`、`encoding_inputs.json`：用途、图序与实际输入尺寸。
- `scene.zip`：项目格式的可导入便携场景。
- `result-cfg-1.json`、`result-cfg-3.json`、`verification.json`：运行与验证记录。

双人实验目录：`/home/lipz/T8Qwen/Comfyui-Qwen-Image-2.1-MultiAngle-T8/.local/qwen21-two-person-interaction`。

- `README.md`、`run_interaction.py`：执行说明与脚本。
- `people_prompts.json`、`interaction_prompt.txt`：人物生成与互动的完整原始提示词。
- `scene.json`、`manifest.json`、`scene.zip`：参考配置、图序与便携场景。
- `result-seed-2026101180-cfg-3.json`、`verification.json`：运行与验证记录。

对比目录证据：本目录的 `README.md`、`src/image_nodes.py`、`src/model_nodes.py`、`src/generate_assets.py` 以及 `runs/001-autumn-lounge/report.json`。本文只归档和对比这些已有实验，没有改动既有工作流或结果图片。

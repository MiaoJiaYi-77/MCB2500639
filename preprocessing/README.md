# preprocessing 目录说明

本目录包含数据预处理相关的脚本，用于准备用于训练与评估的图像数据（包括增强、划分、特征提取、超像素预处理以及构建图/数据集等）。下面对每个文件/脚本的目的、输入/输出、使用示例和注意事项做简明说明，便于快速上手与二次开发。

## 目录结构（主要文件）
- `build_graph.py`：把图像/超像素或其他单元构建成图结构（nodes & edges）。通常用于基于图的特征或图神经网络前处理。
- `create_dataset.py`：把原始图像和标注转换成训练/验证/测试集所需的格式（例如生成 YOLO/检测/分类所需的数据组织）。
- `extract_features.py`：从图像或分割结果中提取特征（颜色直方图、纹理特征、预训练网络特征等），并将特征保存为可加载的文件。
- `image_enhancement.py`：图像增强与预处理工具（例如直方图均衡、gamma 校正、噪声去除、对比度/亮度调整等），用于提升模型训练的鲁棒性。
- `preprocess_superpixel.py`：超像素（superpixel）相关的预处理脚本，可能包括生成超像素、合并/过滤、以及把超像素映射回原图的辅助操作。
- `run_preprocessing.py`：预处理流水线入口脚本，按顺序运行上述模块组成的处理流程（可能读取配置文件或命令行参数）。

> 项目根目录下通常还包含 `args.yaml`、`data.yaml` 等配置文件供脚本读取（例如数据路径、增强参数、超像素参数、输出目录等）。

## 常见输入 / 输出
- 输入（Input）
  - 原始图像文件夹（例如 `数据集3713/images/train/` 等）
  - 标注文件（例如 `数据集3713/labels/*.txt`，YOLO 格式或自定义格式）
  - 配置文件（例如 `args.yaml`, `data.yaml`）
- 输出（Output）
  - 处理后保存的图像、增强图像或中间结果（通常在 `preprocessed` 或自定义输出目录）
  - 特征文件（如 `.npy`, `.pkl` 等）
  - 生成的数据集清单 / 划分文件（train/val/test）
  - 图结构文件（如 GraphML, edge list, 或自定义格式）

## 快速使用示例
下面给出常见的运行方式示例（假定在项目根目录下，并且使用 Python 环境）。请根据实际脚本中定义的参数调整命令行参数或配置文件内容。

1) 运行整个预处理流水线（若 `run_preprocessing.py` 支持配置文件）

```powershell
python preprocessing/run_preprocessing.py --config args.yaml
```

2) 单独运行图像增强脚本

```powershell
python preprocessing/image_enhancement.py --input 数据集3713/images/train --output preprocessed/images/train --mode augment
```

3) 生成数据集划分

```powershell
python preprocessing/create_dataset.py --data-config data.yaml --out-dir datasets/processed
```

4) 提取特征

```powershell
python preprocessing/extract_features.py --images datasets/processed/images --out features/features.npy
```

注：以上参数名是示例。请查看各脚本顶部或 `--help` 输出以获取精确参数名称：

```powershell
python preprocessing/image_enhancement.py --help
```

## 依赖（建议）
下面是常见的 Python 库依赖，供创建虚拟环境与安装时参考：

- Python >= 3.8
- numpy
- opencv-python (cv2)
- scikit-image
- scikit-learn
- Pillow
- tqdm
- networkx (如果需要构建/保存图)

安装示例：

```powershell
pip install -r requirements.txt
# 或者手动安装
pip install numpy opencv-python scikit-image scikit-learn pillow tqdm networkx
```

如果项目根目录没有 `requirements.txt`，建议将上述依赖写入一个并提交。

## 输入/输出契约（简短契约说明）
- 输入：一组图像与对应标注（可选择性包含配置文件），路径与格式需与 `data.yaml`/`args.yaml` 匹配。
- 输出：用于训练/评估的目录结构（已增强图像、划分的 train/val/test、特征文件、图文件）。
- 错误模式：路径不存在、标注格式不符、磁盘空间不足或依赖缺失会导致失败；脚本应在这些情况下给出易读的错误信息。

## 常见边界/注意事项
- 标注文件格式不一致（例如 YOLO vs VOC）：在运行前确认 `create_dataset.py` 所需的标注格式或实现转换。
- 大数据集内存：特征提取或一次性读入所有图像可能会占用大量内存，建议逐步处理并把结果写入磁盘。
- 随机性：数据增强与划分应提供随机种子参数以保证可复现性。

## 建议的改进（可选）
- 为每个脚本实现 `--dry-run` 和 `--verbose` 模式，以便调试。
- 给 `run_preprocessing.py` 增加配置文件解析（YAML/JSON）并支持多阶段开关（只做增强 / 只做划分 / 只提取特征）。
- 添加单元测试（例如对小样本集运行并断言输出文件存在）。
- 添加 `requirements.txt` 或 `pyproject.toml` 来锁定依赖版本。
- 把常用 I/O 路径与参数抽象成一个配置类，便于在训练/评估脚本中共享。

## 故障排查小贴士
- 如果脚本报找不到模块：请确认已激活虚拟环境并安装依赖。
- 如果路径报错：检查 `data.yaml` 或传入的 `--input/--output` 路径是否正确（相对/绝对路径）
- 权限/Windows 路径注意：在 Windows 下确保路径中的转义与权限问题，推荐使用原始字符串或正斜杠 `/`。

---

如果你希望，我可以：
- 打开并阅读每个脚本，依据文件内部的真实参数把 README 示例命令精确化；
- 为 `preprocessing` 增加一个小型测试（比如 `tests/test_preprocessing_pipeline.py`）来验证关键步骤；
- 生成 `requirements.txt`（基于脚本中实际导入的库）。

请选择下一步（例如“请把命令参数具体化”或“生成 requirements.txt”），我会继续按需修改与验证。
"""
图数据集创建脚本
整合超像素、特征和图结构,创建PyTorch Geometric Dataset
"""
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import argparse
import os


class ContainerGraphDataset(Dataset):
    """集装箱图数据集 (PyTorch Geometric格式)"""
    
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        """
        初始化数据集
        
        参数:
            root: 数据根目录
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据变换
            pre_transform: 预处理变换
        """
        self.split = split
        self.root_path = Path(root)
        # 如果用户或外部调用时拼写错误（例如 preproccessed vs preprocessed），尝试自动修正常见拼写错误
        if not self.root_path.exists():
            as_str = str(self.root_path)
            # 常见错误：preproccessed -> preprocessed
            if 'preproccessed' in as_str.lower():
                alt = Path(as_str.lower().replace('preproccessed', 'preprocessed'))
                if alt.exists():
                    print(f"⚠️ 检测到预处理目录拼写疑似错误: {self.root_path}\n    已自动切换到: {alt}")
                    self.root_path = alt
            # 额外尝试：如果 root 的父目录中存在 'preprocessed'，则使用之
            if not self.root_path.exists():
                parent = self.root_path.parent
                cand = parent / 'preprocessed'
                if cand.exists():
                    print(f"⚠️ 预处理根目录不存在: {self.root_path}\n    使用检测到的目录: {cand}")
                    self.root_path = cand
        
        # 数据路径
        self.superpixel_dir = self.root_path / 'superpixels' / split
        self.features_dir = self.root_path / 'node_features' / split
        self.graphs_dir = self.root_path / 'graphs' / split
        self.labels_dir = Path(root).parent / 'data' / 'labels' / split
        
        # 获取所有图像名称
        self.image_names = self._get_image_names()
        
        super(ContainerGraphDataset, self).__init__(root, transform, pre_transform)
    
    def _get_image_names(self):
        """获取所有图像名称"""
        # 从特征文件获取图像名称
        # 如果特征目录不存在或为空，返回空列表（上层代码会友好处理空数据集）
        if not self.features_dir.exists():
            print(f"⚠️ 特征目录不存在: {self.features_dir}，将返回空数据集。请确认已运行特征提取脚本（preprocess/extract_features.py）。")
            return []

        feature_files = sorted(list(self.features_dir.glob('*_features.npy')))
        if len(feature_files) == 0:
            print(f"⚠️ 特征目录为空: {self.features_dir}，将返回空数据集。")
            return []

        image_names = [f.stem.replace('_features', '') for f in feature_files]

        return image_names
    
    @property
    def raw_file_names(self):
        """原始文件名列表"""
        return []
    
    @property
    def processed_file_names(self):
        """处理后的文件名列表"""
        return [f'{name}.pt' for name in self.image_names]
    
    def download(self):
        """下载数据(不需要)"""
        pass
    
    def process(self):
        """处理数据并保存"""
        print(f"处理 {self.split} 集数据...")
        
        for idx, image_name in enumerate(tqdm(self.image_names, desc=f"处理{self.split}集")):
            try:
                # 加载图数据
                data = self._load_graph_data(image_name)
                
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # 保存处理后的数据
                torch.save(data, self.processed_paths[idx])
                
            except Exception as e:
                print(f"\n处理失败 {image_name}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def _load_graph_data(self, image_name):
        """
        加载单个图数据
        
        参数:
            image_name: 图像名称
            
        返回:
            data: PyG Data对象
        """
        # 1. 加载节点特征
        # 优先使用预处理后的特征文件 *_features_proc.npy（若存在），否则回退到原始 *_features.npy
        proc_path = self.features_dir / f"{image_name}_features_proc.npy"
        raw_path = self.features_dir / f"{image_name}_features.npy"
        if proc_path.exists():
            node_features = np.load(proc_path)
        elif raw_path.exists():
            node_features = np.load(raw_path)
        else:
            raise FileNotFoundError(f"找不到节点特征文件: {proc_path} 或 {raw_path}")
        x = torch.from_numpy(node_features).float()
        
        # 2. 加载图结构
        edge_index_path = self.graphs_dir / f"{image_name}_edge_index.npy"
        edge_index = np.load(edge_index_path)
        edge_index = torch.from_numpy(edge_index).long()
        
        # 3. 加载边属性(可选)
        edge_attr_path = self.graphs_dir / f"{image_name}_edge_attr.npy"
        if edge_attr_path.exists():
            edge_attr = np.load(edge_attr_path)
            edge_attr = torch.from_numpy(edge_attr).float().unsqueeze(1)  # (E, 1)
        else:
            edge_attr = None
        
        # 4. 加载图级别标签
        graph_label = self._load_graph_label(image_name)
        
        # 5. 加载节点级别标签(用于分割任务)
        node_labels = self._load_node_labels(image_name)
        
        # 6. 加载超像素信息(用于后处理)
        sp_info_path = self.superpixel_dir / f"{image_name}_info.json"
        with open(sp_info_path, 'r', encoding='utf-8') as f:
            sp_info = json.load(f)
        
        # 创建Data对象
        data = Data(
            x=x,                           # 节点特征 (N, 2048)
            edge_index=edge_index,         # 边索引 (2, E)
            edge_attr=edge_attr,           # 边属性 (E, 1)
            y=graph_label,                 # 图级别标签(用于分类)
            node_y=node_labels,            # 节点级别标签(用于分割)
            image_name=image_name,         # 图像名称
            num_superpixels=sp_info['num_superpixels']  # 超像素数量
        )
        
        return data
    
    def _load_graph_label(self, image_name):
        """
        加载图级别标签(有残损/无残损)
        
        参数:
            image_name: 图像名称
            
        返回:
            label: 0=无残损, 1=有残损
        """
        label_file = self.labels_dir / f"{image_name}.txt"
        
        if not label_file.exists():
            # 如果标签文件不存在,返回-1(未标注)
            return torch.tensor(-1, dtype=torch.long)
        
        # 读取YOLO标签文件
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # 如果有任何标注框,说明有残损
        if len(lines) > 0:
            return torch.tensor(1, dtype=torch.long)  # 有残损
        else:
            return torch.tensor(0, dtype=torch.long)  # 无残损
    
    def _load_node_labels(self, image_name):
        """
        加载节点级别标签(将YOLO标签映射到超像素)
        
        参数:
            image_name: 图像名称
            
        返回:
            node_labels: 节点标签 (N,)
        """
        # 加载超像素分割
        segments_path = self.superpixel_dir / f"{image_name}_segments.npy"
        segments = np.load(segments_path)
        
        # 加载超像素信息
        sp_info_path = self.superpixel_dir / f"{image_name}_info.json"
        with open(sp_info_path, 'r', encoding='utf-8') as f:
            sp_info = json.load(f)
        
        num_superpixels = sp_info['num_superpixels']
        
        # 初始化节点标签(0=背景/正常)
        node_labels = np.zeros(num_superpixels, dtype=np.int64)
        
        # 加载YOLO标签
        label_file = self.labels_dir / f"{image_name}.txt"
        
        if not label_file.exists():
            # 没有标签文件,所有节点都是背景
            return torch.from_numpy(node_labels).long()
        
        # 读取YOLO标签
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            # 空文件,所有节点都是背景
            return torch.from_numpy(node_labels).long()
        
        # 图像尺寸
        img_h, img_w = segments.shape
        
        # 处理每个标注框
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 转换为像素坐标
            x_center_px = int(x_center * img_w)
            y_center_px = int(y_center * img_h)
            width_px = int(width * img_w)
            height_px = int(height * img_h)
            
            # 计算边界框
            x1 = max(0, x_center_px - width_px // 2)
            y1 = max(0, y_center_px - height_px // 2)
            x2 = min(img_w - 1, x_center_px + width_px // 2)
            y2 = min(img_h - 1, y_center_px + height_px // 2)
            
            # 找到与该边界框重叠的超像素
            bbox_region = segments[y1:y2, x1:x2]
            
            if bbox_region.size > 0:
                # 获取该区域内的所有超像素ID
                unique_sp_ids = np.unique(bbox_region)
                
                # 标记这些超像素为对应的类别(class_id + 1,因为0是背景)
                for sp_id in unique_sp_ids:
                    node_labels[sp_id] = class_id + 1
        
        return torch.from_numpy(node_labels).long()
    
    def len(self):
        """数据集大小"""
        return len(self.image_names)
    
    def get(self, idx):
        """获取单个数据"""
        # PyTorch 2.6+需要设置weights_only=False以加载PyG的Data对象
        data = torch.load(self.processed_paths[idx], weights_only=False)
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


def create_datasets(root, splits=['train', 'test']):
    """
    创建所有数据集
    
    参数:
        root: 数据根目录
        splits: 数据集分割列表 (默认: ['train', 'test'], 验证集公布后可改为 ['train', 'val', 'test'])
        
    返回:
        datasets: 数据集字典
    """
    print("=" * 60)
    print("创建图数据集")
    print("=" * 60)
    print(f"数据根目录: {root}")
    print(f"数据集分割: {splits}")
    print("=" * 60)
    
    datasets = {}
    
    for split in splits:
        print(f"\n创建 {split} 集...")
        try:
            dataset = ContainerGraphDataset(root=root, split=split)
            datasets[split] = dataset
            
            print(f"✓ {split} 集创建完成:")
            print(f"  图数量: {len(dataset)}")
            
            # 显示第一个样本的信息
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  样本示例:")
                print(f"    节点特征: {sample.x.shape}")
                print(f"    边索引: {sample.edge_index.shape}")
                print(f"    图标签: {sample.y.item()}")
                print(f"    节点标签: {sample.node_y.shape}")
                
        except Exception as e:
            print(f"✗ {split} 集创建失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ 数据集创建完成!")
    print("=" * 60)
    
    return datasets


def analyze_dataset(dataset):
    """
    分析数据集统计信息
    
    参数:
        dataset: 图数据集
    """
    print(f"\n数据集分析: {dataset.split}")
    print("=" * 60)
    
    num_graphs = len(dataset)
    num_nodes_list = []
    num_edges_list = []
    num_features = 0
    graph_labels = []
    
    for i in tqdm(range(num_graphs), desc="分析数据集"):
        data = dataset[i]
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)
        num_features = data.num_node_features
        graph_labels.append(data.y.item())
    
    # 统计图标签分布
    graph_labels = np.array(graph_labels)
    unique_labels, counts = np.unique(graph_labels, return_counts=True)
    
    print(f"\n图统计:")
    print(f"  总图数: {num_graphs}")
    print(f"  平均节点数: {np.mean(num_nodes_list):.1f}")
    print(f"  平均边数: {np.mean(num_edges_list):.1f}")
    print(f"  节点特征维度: {num_features}")
    
    print(f"\n图标签分布:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  未标注: {count} ({count/num_graphs*100:.1f}%)")
        elif label == 0:
            print(f"  无残损: {count} ({count/num_graphs*100:.1f}%)")
        elif label == 1:
            print(f"  有残损: {count} ({count/num_graphs*100:.1f}%)")
    
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="创建并分析 PyG 图数据集")
    parser.add_argument('--preprocessed-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='预处理根目录（包含 superpixels/node_features/graphs）')
    parser.add_argument('--split', type=str, default=None, help='单个分割名，如 test_noisy')
    parser.add_argument('--splits', type=str, default=None, help='多个分割名，逗号分隔')
    parser.add_argument('--analyze', action='store_true', help='是否输出数据集统计分析')

    args = parser.parse_args()

    # 解析 splits（默认包含 train/val/test）
    splits = ['train', 'val', 'test']
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    datasets = create_datasets(Path(args.preprocessed_root), splits=splits)

    if args.analyze:
        for split, dataset in datasets.items():
            if dataset is not None:
                analyze_dataset(dataset)

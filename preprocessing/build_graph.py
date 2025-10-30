"""
图边构建脚本
使用KNN算法基于超像素的空间位置构建图的边
"""
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class GraphBuilder:
    """图构建器"""
    
    def __init__(self, k_neighbors=8, include_self=False):
        """
        初始化图构建器
        
        参数:
            k_neighbors: 每个节点连接的邻居数量
            include_self: 是否包含自环边
        """
        self.k_neighbors = k_neighbors
        self.include_self = include_self
        
    def build_edges_knn(self, centroids):
        """
        使用KNN算法构建边
        
        参数:
            centroids: 超像素中心坐标数组 (N, 2)
            
        返回:
            edge_index: 边索引 (2, E)，PyTorch Geometric格式
            edge_attr: 边属性（距离） (E,)
        """
        num_nodes = len(centroids)
        
        # 如果节点数少于k+1，调整k值
        k = min(self.k_neighbors, num_nodes - 1)
        
        if k <= 0:
            # 如果只有一个节点，返回空边
            return np.array([[], []], dtype=np.int64), np.array([], dtype=np.float32)
        
        # 使用sklearn的KNN算法
        # k+1是因为最近的邻居包括自己
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.kneighbors(centroids)
        
        # 构建边列表
        edge_list = []
        edge_distances = []
        
        for i in range(num_nodes):
            # indices[i, 0]是节点自己，从indices[i, 1:]开始是真正的邻居
            start_idx = 0 if self.include_self else 1
            
            for j in range(start_idx, len(indices[i])):
                neighbor = indices[i][j]
                distance = distances[i][j]
                
                # 添加双向边（无向图）
                edge_list.append([i, neighbor])
                edge_distances.append(distance)
        
        # 转换为PyTorch Geometric格式
        if len(edge_list) == 0:
            edge_index = np.array([[], []], dtype=np.int64)
            edge_attr = np.array([], dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T  # (2, E)
            edge_attr = np.array(edge_distances, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def build_edges_radius(self, centroids, radius=50.0):
        """
        基于半径的边构建方法（备选）
        
        参数:
            centroids: 超像素中心坐标数组 (N, 2)
            radius: 连接半径
            
        返回:
            edge_index: 边索引 (2, E)
            edge_attr: 边属性（距离） (E,)
        """
        num_nodes = len(centroids)
        
        # 使用sklearn的radius neighbors
        nbrs = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(centroids)
        distances, indices = nbrs.radius_neighbors(centroids)
        
        edge_list = []
        edge_distances = []
        
        for i in range(num_nodes):
            for j, neighbor in enumerate(indices[i]):
                if neighbor != i:  # 排除自环
                    edge_list.append([i, neighbor])
                    edge_distances.append(distances[i][j])
        
        if len(edge_list) == 0:
            edge_index = np.array([[], []], dtype=np.int64)
            edge_attr = np.array([], dtype=np.float32)
        else:
            edge_index = np.array(edge_list, dtype=np.int64).T
            edge_attr = np.array(edge_distances, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def compute_graph_stats(self, edge_index, num_nodes):
        """
        计算图的统计信息
        
        参数:
            edge_index: 边索引 (2, E)
            num_nodes: 节点数量
            
        返回:
            stats: 统计信息字典
        """
        if edge_index.shape[1] == 0:
            return {
                'num_nodes': num_nodes,
                'num_edges': 0,
                'avg_degree': 0.0,
                'max_degree': 0,
                'min_degree': 0
            }
        
        # 计算每个节点的度
        degrees = np.zeros(num_nodes, dtype=np.int32)
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            degrees[src] += 1
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': edge_index.shape[1],
            'avg_degree': float(degrees.mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min())
        }
        
        return stats


def process_dataset_graphs(superpixel_root, output_root, k_neighbors=8, include_self=False, splits=None):
    """
    为整个数据集构建图
    
    参数:
        superpixel_root: 超像素数据根目录
        output_root: 输出根目录
        k_neighbors: KNN邻居数量
        include_self: 是否包含自环
    """
    print("=" * 60)
    print("图边构建")
    print("=" * 60)
    print(f"超像素路径: {superpixel_root}")
    print(f"输出路径: {output_root}")
    print(f"KNN邻居数: {k_neighbors}")
    print(f"包含自环: {include_self}")
    print("=" * 60)
    
    superpixel_path = Path(superpixel_root)
    output_path = Path(output_root)
    
    # 创建输出目录
    graphs_dir = output_path / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化图构建器
    builder = GraphBuilder(k_neighbors=k_neighbors, include_self=include_self)
    
    # 处理每个数据集分割
    # 默认处理 train/val/test；可通过 --split/--splits 指定（例如 test_noisy）
    if splits is None or len(splits) == 0:
        splits = ['train', 'val', 'test']
    
    for split in splits:
        sp_dir = superpixel_path / 'superpixels' / split
        
        if not sp_dir.exists():
            print(f"\n警告: {split} 目录不存在，跳过")
            continue
        
        # 创建输出子目录
        split_graphs_dir = graphs_dir / split
        split_graphs_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有超像素信息文件
        info_files = sorted(list(sp_dir.glob('*_info.json')))
        
        print(f"\n处理 {split} 集: {len(info_files)} 张图像")
        
        # 统计信息
        stats = {
            'total_images': len(info_files),
            'processed': 0,
            'failed': 0,
            'avg_nodes': 0.0,
            'avg_edges': 0.0,
            'avg_degree': 0.0,
            'k_neighbors': k_neighbors
        }
        
        total_nodes = 0
        total_edges = 0
        total_degree = 0.0
        
        # 处理每张图像
        for info_file in tqdm(info_files, desc=f"构建{split}集图"):
            try:
                # 获取图像名称
                image_name = info_file.stem.replace('_info', '')
                
                # 加载超像素信息
                with open(info_file, 'r', encoding='utf-8') as f:
                    sp_info = json.load(f)
                
                # 提取中心坐标
                centroids = []
                for sp in sp_info['superpixels']:
                    centroids.append(sp['centroid'])
                
                centroids = np.array(centroids, dtype=np.float32)
                num_nodes = len(centroids)
                
                # 构建边
                edge_index, edge_attr = builder.build_edges_knn(centroids)
                
                # 计算图统计信息
                graph_stats = builder.compute_graph_stats(edge_index, num_nodes)
                
                # 保存边索引
                edge_file = split_graphs_dir / f"{image_name}_edge_index.npy"
                np.save(edge_file, edge_index)
                
                # 保存边属性（距离）
                edge_attr_file = split_graphs_dir / f"{image_name}_edge_attr.npy"
                np.save(edge_attr_file, edge_attr)
                
                # 保存图统计信息
                graph_info = {
                    'image_name': image_name,
                    'num_nodes': num_nodes,
                    'num_edges': int(edge_index.shape[1]),
                    'avg_degree': graph_stats['avg_degree'],
                    'max_degree': graph_stats['max_degree'],
                    'min_degree': graph_stats['min_degree']
                }
                
                graph_info_file = split_graphs_dir / f"{image_name}_graph_info.json"
                with open(graph_info_file, 'w', encoding='utf-8') as f:
                    json.dump(graph_info, f, indent=2, ensure_ascii=False)
                
                stats['processed'] += 1
                total_nodes += num_nodes
                total_edges += edge_index.shape[1]
                total_degree += graph_stats['avg_degree']
                
            except Exception as e:
                print(f"\n处理失败 {info_file.name}: {str(e)}")
                stats['failed'] += 1
                import traceback
                traceback.print_exc()
        
        # 计算平均统计
        if stats['processed'] > 0:
            stats['avg_nodes'] = total_nodes / stats['processed']
            stats['avg_edges'] = total_edges / stats['processed']
            stats['avg_degree'] = total_degree / stats['processed']
        
        # 打印统计信息
        print(f"\n{split} 集处理完成:")
        print(f"  成功处理: {stats['processed']} 张")
        print(f"  处理失败: {stats['failed']} 张")
        print(f"  平均节点数: {stats['avg_nodes']:.1f}")
        print(f"  平均边数: {stats['avg_edges']:.1f}")
        print(f"  平均度: {stats['avg_degree']:.2f}")
        
        # 保存统计信息
        stats_path = split_graphs_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✓ 图边构建完成!")
    print(f"\n输出目录结构:")
    print(f"  {output_path}/")
    print(f"    └── graphs/")
    print(f"        ├── train/")
    print(f"        │   ├── *_edge_index.npy (边索引 2×E)")
    print(f"        │   ├── *_edge_attr.npy (边属性-距离)")
    print(f"        │   ├── *_graph_info.json (图统计信息)")
    print(f"        │   └── stats.json")
    print(f"        ├── val/")
    print(f"        └── test/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于超像素中心的KNN图构建")
    parser.add_argument('--preprocessed-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='预处理根目录（包含 superpixels/graphs 等）')
    parser.add_argument('--k-neighbors', type=int, default=8, help='KNN 邻居数量')
    parser.add_argument('--include-self', action='store_true', help='是否包含自环')
    parser.add_argument('--split', type=str, default=None, help='单个分割名，如 test_noisy')
    parser.add_argument('--splits', type=str, default=None, help='多个分割名，逗号分隔')

    args = parser.parse_args()

    # 解析 splits
    splits = None
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    process_dataset_graphs(
        superpixel_root=Path(args.preprocessed_root),
        output_root=Path(args.preprocessed_root),
        k_neighbors=args.k_neighbors,
        include_self=args.include_self,
        splits=splits
    )

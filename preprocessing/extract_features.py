"""
节点特征提取脚本（高性能优化版）
针对RTX 4050 6GB + 16核CPU优化
- 多进程数据预加载
- 批量化特征提取
- 向量化超像素处理
- 显存动态管理
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import warnings
import gc
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """获取GPU显存信息（MB）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': total - reserved,
            'total': total
        }
    return None


def clear_gpu_memory():
    """清理GPU显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class SuperpixelDataset(Dataset):
    """超像素数据集，优化的批量加载"""
    
    def __init__(self, image_files, segment_files, info_files, transform=None):
        self.image_files = image_files
        self.segment_files = segment_files
        self.info_files = info_files
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        image = cv2.imread(str(self.image_files[idx]))
        if image is None:
            raise ValueError(f"无法读取图像: {self.image_files[idx]}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 加载超像素数据（延迟加载，节省内存）
        segments = np.load(self.segment_files[idx])
        
        with open(self.info_files[idx], 'r', encoding='utf-8') as f:
            sp_info = json.load(f)
        
        # 图像预处理
        if self.transform:
            image_tensor = self.transform(image_rgb)
        
        return {
            'image': image_tensor,
            'segments': segments,
            'sp_info': sp_info,
            'filename': self.image_files[idx].stem
        }


class FeatureExtractor:
    """ResNet特征提取器（高性能优化版）"""
    
    def __init__(self, model_name='resnet50', device='cuda', use_amp=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and torch.cuda.is_available()
        
        logger.info(f"使用设备: {self.device}")
        if self.use_amp:
            logger.info("✓ 混合精度训练已启用")
        
        # 显示GPU信息
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            logger.info(f"GPU显存: {gpu_info['total']:.1f}MB 总计, {gpu_info['free']:.1f}MB 可用")
        
        # 加载预训练模型
        logger.info(f"加载模型: {model_name}...")
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            weights = models.ResNet101_Weights.IMAGENET1K_V1
            self.model = models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 移除最后的全连接层和pooling层
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 禁用梯度计算，节省显存
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("✓ 模型加载完成")
        
        # 预热模型
        self._warmup()
    
    def _warmup(self):
        """预热模型，优化后续推理速度"""
        logger.info("预热模型...")
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    _ = self.model(dummy_input)
            else:
                _ = self.model(dummy_input)
        clear_gpu_memory()
        logger.info("✓ 模型预热完成")
    
    @torch.no_grad()
    def extract_batch_features(self, images):
        """批量提取图像特征"""
        images = images.to(self.device)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                features = self.model(images)
        else:
            features = self.model(images)
        
        return features
    
    def extract_superpixel_features_vectorized(self, feature_map, segments, sp_info):
        """
        向量化超像素特征提取（关键优化点）
        
        参数:
            feature_map: 特征图 (C, H_f, W_f)
            segments: 超像素分割 (H, W)
            sp_info: 超像素信息
            
        返回:
            node_features: (N, C)
        """
        C, H_f, W_f = feature_map.shape
        H_orig, W_orig = segments.shape
        
        # 计算缩放比例
        scale_h = H_f / H_orig
        scale_w = W_f / W_orig
        
        num_superpixels = sp_info['num_superpixels']
        
        # 将segments缩放到特征图尺寸（使用最近邻插值）
        segments_resized = cv2.resize(
            segments.astype(np.float32), 
            (W_f, H_f), 
            interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        
        # 向量化提取：为每个超像素ID计算特征
        node_features = np.zeros((num_superpixels, C), dtype=np.float32)
        
        # 重塑特征图为 (C, H_f*W_f)
        feature_map_flat = feature_map.reshape(C, -1)
        segments_flat = segments_resized.flatten()
        
        # 对每个超像素ID进行批量处理
        unique_ids = np.unique(segments_flat)
        for sp_id in unique_ids:
            if sp_id < 0 or sp_id >= num_superpixels:
                continue
            
            # 找到属于当前超像素的所有位置
            mask = (segments_flat == sp_id)
            
            if mask.sum() == 0:
                continue
            
            # 提取并平均这些位置的特征
            sp_features = feature_map_flat[:, mask]  # (C, N_pixels)
            node_features[sp_id] = sp_features.mean(axis=1)
        
        return node_features
    
    def get_feature_dim(self):
        """获取特征维度"""
        return 2048  # ResNet50 layer4


def collate_fn(batch):
    """自定义collate函数，处理可变大小的数据"""
    return batch


def process_single_image(item, extractor, output_dir):
    """
    处理单张图像（供多进程使用）
    
    返回:
        success: 是否成功
        info: 处理信息
    """
    try:
        filename = item['filename']
        image = item['image']
        segments = item['segments']
        sp_info = item['sp_info']
        
        # 扩展batch维度
        image_batch = image.unsqueeze(0)
        
        # 提取特征图
        feature_map = extractor.extract_batch_features(image_batch)
        feature_map = feature_map.squeeze(0).cpu().numpy()  # (C, H_f, W_f)
        
        # 向量化提取超像素特征
        node_features = extractor.extract_superpixel_features_vectorized(
            feature_map, segments, sp_info
        )
        
        # 保存特征
        feature_file = output_dir / f"{filename}_features.npy"
        np.save(feature_file, node_features)
        
        return True, {
            'name': filename,
            'num_superpixels': sp_info['num_superpixels']
        }
        
    except Exception as e:
        return False, {
            'name': item.get('filename', 'unknown'),
            'error': str(e)
        }


def estimate_optimal_batch_size(device, model):
    """智能估算最优batch size"""
    if not torch.cuda.is_available():
        return 1
    
    logger.info("智能估算最优batch size...")
    clear_gpu_memory()
    
    # 测试不同batch size的性能
    test_sizes = [1, 2, 4, 8, 16, 32]
    valid_sizes = []
    
    for bs in test_sizes:
        try:
            test_input = torch.randn(bs, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(test_input)
            valid_sizes.append(bs)
            clear_gpu_memory()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                clear_gpu_memory()
                break
            raise e
    
    # 选择最大可用batch size的75%作为安全值
    if valid_sizes:
        optimal = max(1, int(max(valid_sizes) * 0.75))
    else:
        optimal = 1
    
    logger.info(f"✓ 最优batch size: {optimal}")
    return optimal


def process_dataset_features(dataset_root, superpixel_root, output_root, 
                             model_name='resnet50', device='cuda', 
                             use_amp=True, batch_size=None,
                             num_workers=8, splits=None):
    """
    为整个数据集提取特征（高性能优化版）
    
    关键优化：
    1. 多进程数据加载 (DataLoader with num_workers)
    2. 批量特征提取
    3. 向量化超像素处理
    4. 预加载和pin_memory
    """
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("节点特征提取（高性能优化版）")
    logger.info("=" * 60)
    logger.info(f"数据集路径: {dataset_root}")
    logger.info(f"超像素路径: {superpixel_root}")
    logger.info(f"输出路径: {output_root}")
    logger.info(f"CPU核心数: {mp.cpu_count()}, 使用workers: {num_workers}")
    logger.info("=" * 60)
    
    dataset_path = Path(dataset_root)
    superpixel_path = Path(superpixel_root)
    output_path = Path(output_root)
    
    # 创建输出目录
    features_dir = output_path / 'node_features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化特征提取器
    extractor = FeatureExtractor(
        model_name=model_name,
        device=device,
        use_amp=use_amp
    )
    
    # 估算最优batch size
    if batch_size is None:
        batch_size = estimate_optimal_batch_size(extractor.device, extractor.model)
    
    feature_dim = extractor.get_feature_dim()
    logger.info(f"特征维度: {feature_dim}")
    logger.info(f"批处理大小: {batch_size}")
    
    # 处理每个数据集分割
    if splits is None or len(splits) == 0:
        splits = ['train', 'val', 'test']
    
    for split in splits:
        split_start_time = datetime.now()
        images_dir = dataset_path / 'images' / split
        sp_dir = superpixel_path / 'superpixels' / split
        
        if not images_dir.exists() or not sp_dir.exists():
            logger.warning(f"{split} 目录不存在，跳过")
            continue
        
        # 创建输出子目录
        split_features_dir = features_dir / split
        split_features_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有文件路径
        segment_files = sorted(list(sp_dir.glob('*_segments.npy')))
        
        logger.info(f"\n处理 {split} 集: {len(segment_files)} 张图像")
        
        if len(segment_files) == 0:
            logger.warning(f"{split} 集没有图像，跳过")
            continue
        
        # 准备数据路径
        image_files = []
        info_files = []
        
        for seg_file in segment_files:
            image_name = seg_file.stem.replace('_segments', '')
            
            # 查找图像文件
            image_file = None
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                candidate = images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break
            
            if image_file is None:
                continue
            
            info_file = sp_dir / f"{image_name}_info.json"
            if not info_file.exists():
                continue
            
            image_files.append(image_file)
            info_files.append(info_file)
        
        # 创建数据集和数据加载器
        dataset = SuperpixelDataset(
            image_files,
            segment_files[:len(image_files)],
            info_files,
            transform=extractor.transform
        )
        
        # 关键优化：多进程数据加载
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # 统计信息
        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'feature_dim': feature_dim,
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        
        total_superpixels = 0
        failed_images = []
        
        # 批量处理
        progress_bar = tqdm(dataloader, desc=f"提取{split}集特征")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # 批量提取特征
                images = torch.stack([item['image'] for item in batch_data])
                
                # 提取特征图
                feature_maps = extractor.extract_batch_features(images)
                feature_maps = feature_maps.cpu().numpy()  # (B, C, H_f, W_f)
                
                # 处理每张图像的超像素
                for i, item in enumerate(batch_data):
                    try:
                        filename = item['filename']
                        segments = item['segments']
                        sp_info = item['sp_info']
                        feature_map = feature_maps[i]  # (C, H_f, W_f)
                        
                        # 向量化提取超像素特征
                        node_features = extractor.extract_superpixel_features_vectorized(
                            feature_map, segments, sp_info
                        )
                        
                        # 保存特征
                        feature_file = split_features_dir / f"{filename}_features.npy"
                        np.save(feature_file, node_features)
                        
                        stats['processed'] += 1
                        total_superpixels += sp_info['num_superpixels']
                        
                    except Exception as e:
                        logger.error(f"处理失败 {item['filename']}: {e}")
                        stats['failed'] += 1
                        failed_images.append({'name': item['filename'], 'error': str(e)})
                
                # 定期清理显存
                if (batch_idx + 1) % 10 == 0:
                    clear_gpu_memory()
                    
                    # 更新进度条
                    if torch.cuda.is_available():
                        gpu_info = get_gpu_memory_info()
                        progress_bar.set_postfix({
                            'GPU': f"{gpu_info['allocated']:.0f}MB",
                            '成功': stats['processed'],
                            '失败': stats['failed']
                        })
                
            except Exception as e:
                logger.error(f"批次处理失败: {e}")
                stats['failed'] += len(batch_data)
                clear_gpu_memory()
        
        # 计算统计信息
        if stats['processed'] > 0:
            stats['avg_superpixels'] = total_superpixels / stats['processed']
        
        split_duration = (datetime.now() - split_start_time).total_seconds()
        stats['processing_time_seconds'] = split_duration
        stats['images_per_second'] = stats['processed'] / split_duration if split_duration > 0 else 0
        stats['failed_images'] = failed_images
        
        # 打印统计信息
        logger.info(f"\n{split} 集处理完成:")
        logger.info(f"  成功处理: {stats['processed']} 张")
        logger.info(f"  处理失败: {stats['failed']} 张")
        logger.info(f"  平均超像素数量: {stats.get('avg_superpixels', 0):.1f}")
        logger.info(f"  处理时间: {split_duration:.2f} 秒")
        logger.info(f"  处理速度: {stats['images_per_second']:.2f} 张/秒")
        
        # 保存统计信息
        stats_path = split_features_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        clear_gpu_memory()
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 节点特征提取完成!")
    logger.info(f"总处理时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="节点特征提取（高性能优化版）")
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).parent.parent / "data"),
                        help='原始数据根目录')
    parser.add_argument('--preprocessed-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='预处理根目录')
    parser.add_argument('--output-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='输出根目录')
    parser.add_argument('--model-name', type=str, default='resnet50', choices=['resnet50','resnet101'],
                        help='特征提取模型')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='使用混合精度')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='不使用混合精度')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批处理大小（None则自动）')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='数据加载进程数（建议CPU核心数的50-75%）')
    parser.add_argument('--split', type=str, default=None, help='单个分割')
    parser.add_argument('--splits', type=str, default=None, help='多个分割（逗号分隔）')

    args = parser.parse_args()

    # 解析splits
    splits = None
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    process_dataset_features(
        dataset_root=Path(args.data_root),
        superpixel_root=Path(args.preprocessed_root),
        output_root=Path(args.output_root),
        model_name=args.model_name,
        device=args.device,
        use_amp=args.use_amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=splits
    )
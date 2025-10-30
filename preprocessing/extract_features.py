"""
节点特征提取脚本（显存优化版）
使用预训练的ResNet-50提取每个超像素的深度特征
针对6GB显存优化：批量处理、混合精度、显存监控
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

warnings.filterwarnings('ignore')

# 配置日志
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
    """超像素数据集，用于批量加载"""
    
    def __init__(self, image_files, segment_files, info_files, transform=None):
        """
        初始化数据集
        
        参数:
            image_files: 图像文件列表
            segment_files: 超像素分割文件列表
            info_files: 超像素信息文件列表
            transform: 图像预处理转换
        """
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
        original_size = image.shape[:2]
        
        # 加载超像素数据
        segments = np.load(self.segment_files[idx])
        
        with open(self.info_files[idx], 'r', encoding='utf-8') as f:
            sp_info = json.load(f)
        
        # 图像预处理
        if self.transform:
            image_tensor = self.transform(image_rgb)
        else:
            image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image_tensor,
            'segments': segments,
            'sp_info': sp_info,
            'original_size': original_size,
            'filename': self.image_files[idx].stem
        }


class FeatureExtractor:
    """ResNet特征提取器（显存优化版）"""
    
    def __init__(self, model_name='resnet50', layer_name='layer4', device='cuda', 
                 use_amp=True, batch_size=4):
        """
        初始化特征提取器
        
        参数:
            model_name: 模型名称 ('resnet50', 'resnet101')
            layer_name: 提取特征的层 ('layer3', 'layer4')
            device: 计算设备 ('cuda' or 'cpu')
            use_amp: 是否使用混合精度（节省显存）
            batch_size: 批处理大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        
        logger.info(f"使用设备: {self.device}")
        if self.use_amp:
            logger.info("✓ 混合精度训练已启用 (节省显存)")
        
        # 显示GPU信息
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            logger.info(f"GPU显存: {gpu_info['total']:.1f}MB 总计")
            logger.info(f"可用显存: {gpu_info['free']:.1f}MB")
        
        # 加载预训练模型
        logger.info(f"加载模型: {model_name}...")
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 移除最后的全连接层（不需要分类）
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 如果有多GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            logger.info(f"检测到 {torch.cuda.device_count()} 个GPU，使用多GPU并行")
            self.model = nn.DataParallel(self.model)
        
        self.layer_name = layer_name
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ResNet输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("✓ 模型加载完成")
        
    def extract_image_features(self, image_path):
        """
        提取整张图像的特征图
        
        参数:
            image_path: 图像路径
            
        返回:
            feature_map: 特征图 (C, H, W)
            original_size: 原始图像尺寸 (H, W)
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_size = image.shape[:2]  # (H, W)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 预处理
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # 前向传播提取特征
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    feature_map = self.model(image_tensor)
            else:
                feature_map = self.model(image_tensor)
            
            feature_map = feature_map.squeeze(0)  # (C, H_f, W_f)
        
        return feature_map, original_size
    
    def extract_batch_features(self, images):
        """
        批量提取图像特征
        
        参数:
            images: 图像张量 (B, 3, H, W)
            
        返回:
            features: 特征张量 (B, C, H_f, W_f)
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    features = self.model(images)
            else:
                features = self.model(images)
        
        return features
    
    def extract_superpixel_features_batch(self, image_path, segments, superpixel_info):
        """
        为每个超像素提取特征向量（批量优化版）
        
        参数:
            image_path: 图像路径
            segments: 超像素分割数组 (H, W)
            superpixel_info: 超像素信息字典
            
        返回:
            node_features: 节点特征矩阵 (N, C)
        """
        # 提取整张图像的特征图
        feature_map, original_size = self.extract_image_features(image_path)
        feature_map = feature_map.cpu().numpy()  # (C, H_f, W_f)
        
        C, H_f, W_f = feature_map.shape
        H_orig, W_orig = original_size
        
        # 计算特征图和原图的缩放比例
        scale_h = H_f / H_orig
        scale_w = W_f / W_orig
        
        num_superpixels = superpixel_info['num_superpixels']
        node_features = np.zeros((num_superpixels, C), dtype=np.float32)
        
        # 为每个超像素提取特征
        for sp in superpixel_info['superpixels']:
            sp_id = sp['id']
            
            # 获取超像素的mask
            mask = (segments == sp_id)
            
            # 获取超像素覆盖的像素坐标
            coords = np.argwhere(mask)  # (N_pixels, 2) - (y, x)
            
            if len(coords) == 0:
                continue
            
            # 将坐标映射到特征图
            coords_scaled = coords.astype(np.float32)
            coords_scaled[:, 0] = coords_scaled[:, 0] * scale_h  # y坐标
            coords_scaled[:, 1] = coords_scaled[:, 1] * scale_w  # x坐标
            coords_scaled = coords_scaled.astype(np.int32)
            
            # 限制坐标在特征图范围内
            coords_scaled[:, 0] = np.clip(coords_scaled[:, 0], 0, H_f - 1)
            coords_scaled[:, 1] = np.clip(coords_scaled[:, 1], 0, W_f - 1)
            
            # 从特征图中提取对应位置的特征
            features_at_coords = feature_map[:, coords_scaled[:, 0], coords_scaled[:, 1]]  # (C, N_pixels)
            
            # 平均池化
            sp_feature = np.mean(features_at_coords, axis=1)  # (C,)
            
            node_features[sp_id] = sp_feature
        
        return node_features
    
    def get_feature_dim(self):
        """获取特征维度"""
        return 2048  # ResNet layer4输出


def estimate_batch_size(device, model, input_size=(3, 224, 224)):
    """
    自动估算合适的batch size
    
    参数:
        device: 计算设备
        model: 模型
        input_size: 输入尺寸
    
    返回:
        optimal_batch_size: 最优batch size
    """
    if not torch.cuda.is_available():
        return 1
    
    logger.info("正在估算最优batch size...")
    
    batch_size = 1
    max_batch_size = 64
    
    try:
        clear_gpu_memory()
        
        # 二分查找最大可用batch size
        while batch_size < max_batch_size:
            try:
                test_batch = torch.randn(batch_size, *input_size).to(device)
                with torch.no_grad():
                    _ = model(test_batch)
                
                clear_gpu_memory()
                batch_size *= 2
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    clear_gpu_memory()
                    break
                else:
                    raise e
        
        # 使用找到的最大batch size的70%作为安全值
        optimal_batch_size = max(1, batch_size // 2)
        
    except Exception as e:
        logger.warning(f"batch size估算失败: {e}")
        optimal_batch_size = 4
    
    finally:
        clear_gpu_memory()
    
    logger.info(f"✓ 最优batch size: {optimal_batch_size}")
    return optimal_batch_size


def process_dataset_features(dataset_root, superpixel_root, output_root, 
                             model_name='resnet50', layer_name='layer4',
                             device='cuda', use_amp=True, batch_size=None,
                             auto_batch_size=True, splits=None):
    """
    为整个数据集提取特征（显存优化版）
    
    参数:
        dataset_root: 数据集根目录(图像)
        superpixel_root: 超像素数据根目录
        output_root: 输出根目录
        model_name: 模型名称
        layer_name: 提取特征的层
        device: 计算设备
        use_amp: 是否使用混合精度
        batch_size: 批处理大小（None则自动）
        auto_batch_size: 是否自动估算batch size
        splits: 要处理的数据集分割
    """
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("节点特征提取（显存优化版）")
    logger.info("=" * 60)
    logger.info(f"数据集路径: {dataset_root}")
    logger.info(f"超像素路径: {superpixel_root}")
    logger.info(f"输出路径: {output_root}")
    logger.info(f"模型: {model_name}, 层: {layer_name}")
    logger.info("=" * 60)
    
    dataset_path = Path(dataset_root)
    superpixel_path = Path(superpixel_root)
    output_path = Path(output_root)
    
    # 创建输出目录
    features_dir = output_path / 'node_features'
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化特征提取器
    if batch_size is None:
        batch_size = 4  # 默认值
    
    extractor = FeatureExtractor(
        model_name=model_name,
        layer_name=layer_name,
        device=device,
        use_amp=use_amp,
        batch_size=batch_size
    )
    
    # 自动估算batch size
    if auto_batch_size and torch.cuda.is_available():
        optimal_batch_size = estimate_batch_size(
            extractor.device, 
            extractor.model,
            input_size=(3, 224, 224)
        )
        batch_size = optimal_batch_size
        extractor.batch_size = batch_size
    
    feature_dim = extractor.get_feature_dim()
    logger.info(f"\n特征维度: {feature_dim}")
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
        
        # 获取所有超像素数据文件
        segment_files = sorted(list(sp_dir.glob('*_segments.npy')))
        
        logger.info(f"\n处理 {split} 集: {len(segment_files)} 张图像")
        
        if len(segment_files) == 0:
            logger.warning(f"{split} 集没有图像，跳过")
            continue
        
        # 统计信息
        stats = {
            'total_images': len(segment_files),
            'processed': 0,
            'failed': 0,
            'feature_dim': feature_dim,
            'avg_superpixels': 0,
            'batch_size': batch_size,
            'use_amp': use_amp
        }
        
        total_superpixels = 0
        failed_images = []
        
        # 清理显存
        clear_gpu_memory()
        
        # 处理每张图像
        progress_bar = tqdm(segment_files, desc=f"提取{split}集特征")
        for segment_file in progress_bar:
            try:
                # 获取图像名称
                image_name = segment_file.stem.replace('_segments', '')
                
                # 加载超像素数据
                segments = np.load(segment_file)
                
                info_file = sp_dir / f"{image_name}_info.json"
                with open(info_file, 'r', encoding='utf-8') as f:
                    sp_info = json.load(f)
                
                # 获取对应的图像文件，支持多种扩展名
                image_file = None
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
                    candidate = images_dir / f"{image_name}{ext}"
                    if candidate.exists():
                        image_file = candidate
                        break
                
                if image_file is None:
                    logger.warning(f"找不到图像 {image_name}")
                    stats['failed'] += 1
                    failed_images.append({'name': image_name, 'error': '图像文件不存在'})
                    continue
                
                # 提取超像素特征
                node_features = extractor.extract_superpixel_features_batch(
                    image_file, segments, sp_info
                )
                
                # 保存特征
                feature_file = split_features_dir / f"{image_name}_features.npy"
                np.save(feature_file, node_features)
                
                stats['processed'] += 1
                total_superpixels += sp_info['num_superpixels']
                
                # 每处理10张图像清理一次显存
                if stats['processed'] % 10 == 0:
                    clear_gpu_memory()
                    
                    # 更新进度条信息
                    if torch.cuda.is_available():
                        gpu_info = get_gpu_memory_info()
                        progress_bar.set_postfix({
                            'GPU显存': f"{gpu_info['allocated']:.0f}MB",
                            '成功': stats['processed'],
                            '失败': stats['failed']
                        })
                
            except Exception as e:
                logger.error(f"处理失败 {segment_file.name}: {str(e)}")
                stats['failed'] += 1
                failed_images.append({'name': segment_file.name, 'error': str(e)})
                clear_gpu_memory()  # 出错时清理显存
        
        # 计算统计信息
        if stats['processed'] > 0:
            stats['avg_superpixels'] = total_superpixels / stats['processed']
        
        split_duration = (datetime.now() - split_start_time).total_seconds()
        stats['processing_time_seconds'] = split_duration
        stats['images_per_second'] = len(segment_files) / split_duration if split_duration > 0 else 0
        stats['failed_images'] = failed_images
        
        # 打印统计信息
        logger.info(f"\n{split} 集处理完成:")
        logger.info(f"  成功处理: {stats['processed']} 张")
        logger.info(f"  处理失败: {stats['failed']} 张")
        logger.info(f"  平均超像素数量: {stats['avg_superpixels']:.1f}")
        logger.info(f"  特征维度: {stats['feature_dim']}")
        logger.info(f"  处理时间: {split_duration:.2f} 秒")
        logger.info(f"  处理速度: {stats['images_per_second']:.2f} 张/秒")
        
        if stats['failed'] > 0:
            logger.warning(f"  失败的图像: {[img['name'] for img in failed_images[:5]]}")
        
        # 保存统计信息
        stats_path = split_features_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 清理显存
        clear_gpu_memory()
    
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 节点特征提取完成!")
    logger.info(f"总处理时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    logger.info(f"\n输出目录结构:")
    logger.info(f"  {output_path}/")
    logger.info(f"    └── node_features/")
    logger.info(f"        ├── train/")
    logger.info(f"        │   ├── *_features.npy (节点特征矩阵 N×{feature_dim})")
    logger.info(f"        │   └── stats.json")
    logger.info(f"        ├── val/")
    logger.info(f"        └── test/")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为超像素提取ResNet特征（显存优化版）")
    parser.add_argument('--data-root', type=str, default=str(Path(__file__).parent.parent / "data"),
                        help='原始数据根目录（images/<split>）')
    parser.add_argument('--preprocessed-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='预处理根目录（superpixels/node_features 等）')
    parser.add_argument('--output-root', type=str, default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='输出根目录（通常与 preprocessed 相同）')
    parser.add_argument('--model-name', type=str, default='resnet50', choices=['resnet50','resnet101'],
                        help='特征提取主干模型')
    parser.add_argument('--layer-name', type=str, default='layer4', choices=['layer3','layer4'],
                        help='提取的层')
    parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='使用混合精度（节省显存）')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='不使用混合精度')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批处理大小（None则自动估算）')
    parser.add_argument('--no-auto-batch', action='store_false', dest='auto_batch_size',
                        help='禁用自动batch size估算')
    parser.add_argument('--split', type=str, default=None, help='单个分割名，如 test_noisy')
    parser.add_argument('--splits', type=str, default=None, help='多个分割名，逗号分隔')

    args = parser.parse_args()

    # 解析 splits
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
        layer_name=args.layer_name,
        device=args.device,
        use_amp=args.use_amp,
        batch_size=args.batch_size,
        auto_batch_size=args.auto_batch_size,
        splits=splits
    )
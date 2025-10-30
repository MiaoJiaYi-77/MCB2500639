"""
超像素分割预处理脚本（多核并行优化版）
使用SLIC算法将集装箱图像转换为超像素表示
支持多进程并行处理，充分利用多核CPU
"""
import os
import cv2
import numpy as np
from pathlib import Path
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
import json
import argparse
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SuperpixelPreprocessor:
    """超像素预处理器"""
    
    def __init__(self, n_segments=500, compactness=10, sigma=1):
        """
        初始化超像素预处理器
        
        参数:
            n_segments: 超像素的数量
            compactness: 紧凑度参数,值越大超像素越方形
            sigma: 高斯平滑的sigma值
        """
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        
    def process_image(self, image_path):
        """
        对单张图像进行超像素分割
        
        参数:
            image_path: 图像路径
            
        返回:
            segments: 超像素分割结果 (H, W)，每个像素的值是其所属的超像素ID
            superpixel_info: 超像素信息字典
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # BGR转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为float类型 (0-1范围)
        image_float = img_as_float(image_rgb)
        
        # 执行SLIC超像素分割
        segments = slic(
            image_float,
            n_segments=self.n_segments,
            compactness=self.compactness,
            sigma=self.sigma,
            start_label=0,  # 超像素ID从0开始
            channel_axis=-1  # 明确指定通道轴
        )
        
        # 提取超像素信息
        superpixel_info = self._extract_superpixel_info(segments, image_rgb)
        
        return segments, superpixel_info
    
    def _extract_superpixel_info(self, segments, image):
        """
        提取超像素的详细信息
        
        参数:
            segments: 超像素分割结果
            image: 原始图像 (RGB)
            
        返回:
            info: 包含每个超像素信息的字典
        """
        info = {
            'num_superpixels': 0,
            'superpixels': []
        }
        
        # 使用regionprops提取每个超像素的属性
        regions = regionprops(segments + 1)  # regionprops要求标签从1开始
        
        for region in regions:
            # 获取超像素的中心坐标
            centroid_y, centroid_x = region.centroid
            
            # 获取超像素的边界框
            min_row, min_col, max_row, max_col = region.bbox
            
            # 获取超像素的面积
            area = region.area
            
            # 计算超像素的平均颜色
            mask = segments == (region.label - 1)
            mean_color = image[mask].mean(axis=0).tolist()
            
            sp_info = {
                'id': region.label - 1,  # 转回0开始的ID
                'centroid': [float(centroid_x), float(centroid_y)],
                'bbox': [int(min_col), int(min_row), int(max_col), int(max_row)],
                'area': int(area),
                'mean_color': mean_color
            }
            
            info['superpixels'].append(sp_info)
        
        info['num_superpixels'] = len(info['superpixels'])
        
        return info
    
    def visualize_superpixels(self, image_path, segments, output_path):
        """
        可视化超像素分割结果
        
        参数:
            image_path: 原始图像路径
            segments: 超像素分割结果
            output_path: 输出图像路径
        """
        # 读取原始图像
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 标记边界
        marked = mark_boundaries(image_rgb, segments, color=(1, 0, 0), mode='thick')
        
        # 转换回uint8并保存
        marked_uint8 = (marked * 255).astype(np.uint8)
        marked_bgr = cv2.cvtColor(marked_uint8, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(output_path), marked_bgr)


def process_single_image(args):
    """
    处理单张图像的工作函数（用于多进程）
    
    参数:
        args: 元组 (image_path, split_superpixel_dir, split_visualization_dir, 
                   n_segments, compactness, sigma, should_visualize, counter, lock)
    
    返回:
        result: 字典，包含处理结果
    """
    (image_path, split_superpixel_dir, split_visualization_dir, 
     n_segments, compactness, sigma, should_visualize, counter, lock) = args
    
    result = {
        'success': False,
        'image_name': image_path.name,
        'num_superpixels': 0,
        'error': None
    }
    
    try:
        # 创建预处理器
        preprocessor = SuperpixelPreprocessor(
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma
        )
        
        # 执行超像素分割
        segments, sp_info = preprocessor.process_image(image_path)
        
        # 保存超像素分割结果
        output_name = image_path.stem
        
        # 保存segments数组
        segments_path = split_superpixel_dir / f"{output_name}_segments.npy"
        np.save(segments_path, segments)
        
        # 保存超像素信息为JSON
        info_path = split_superpixel_dir / f"{output_name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(sp_info, f, indent=2, ensure_ascii=False)
        
        # 可视化（如果需要）
        if should_visualize:
            vis_output_path = split_visualization_dir / f"{output_name}_superpixels.jpg"
            preprocessor.visualize_superpixels(image_path, segments, vis_output_path)
        
        result['success'] = True
        result['num_superpixels'] = sp_info['num_superpixels']
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"处理失败 {image_path.name}: {str(e)}")
    
    # 更新进度（线程安全）
    if lock is not None:
        with lock:
            counter.value += 1
            if counter.value % 10 == 0 or counter.value == 1:
                logger.info(f"进度: {counter.value} 张图像已处理")
    
    return result


def process_dataset(dataset_root, output_root, n_segments=500, compactness=10, 
                    sigma=1, visualize_samples=10, splits=None, n_workers=None):
    """
    处理整个数据集（多进程并行版本）
    
    参数:
        dataset_root: 数据集根目录
        output_root: 输出根目录
        n_segments: 超像素数量
        compactness: 紧凑度参数
        sigma: 高斯平滑参数
        visualize_samples: 可视化的样本数量
        splits: 要处理的数据集分割列表
        n_workers: 工作进程数量，None则自动检测
    """
    start_time = datetime.now()
    
    # 确定工作进程数量
    if n_workers is None:
        n_workers = max(1, cpu_count() - 2)  # 保留2个核心给系统
    
    logger.info("=" * 60)
    logger.info("超像素分割预处理（多核并行版）")
    logger.info("=" * 60)
    logger.info(f"数据集路径: {dataset_root}")
    logger.info(f"输出路径: {output_root}")
    logger.info(f"超像素数量: {n_segments}")
    logger.info(f"紧凑度: {compactness}")
    logger.info(f"Sigma: {sigma}")
    logger.info(f"CPU核心数: {cpu_count()}")
    logger.info(f"工作进程数: {n_workers}")
    logger.info("=" * 60)
    
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    
    # 创建输出目录
    superpixel_dir = output_path / 'superpixels'
    visualization_dir = output_path / 'visualizations'
    superpixel_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    # 默认处理 train/val/test
    if splits is None or len(splits) == 0:
        splits = ['train', 'val', 'test']
    
    # 处理每个数据集分割
    for split in splits:
        split_start_time = datetime.now()
        # 支持两种数据布局：
        # 1) data/images/<split>  （以前的脚本结构）
        # 2) data/<split>/images  （当前仓库的结构）
        images_dir_a = dataset_path / 'images' / split
        images_dir_b = dataset_path / split / 'images'

        if images_dir_a.exists():
            images_dir = images_dir_a
            logger.info(f"使用图像目录: {images_dir} (data/images/<split>)")
        elif images_dir_b.exists():
            images_dir = images_dir_b
            logger.info(f"使用图像目录: {images_dir} (data/<split>/images)")
        else:
            logger.warning(f"{split} 目录不存在，跳过")
            continue
        
        # 创建输出子目录
        split_superpixel_dir = superpixel_dir / split
        split_visualization_dir = visualization_dir / split
        split_superpixel_dir.mkdir(parents=True, exist_ok=True)
        split_visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像
        image_files = sorted(list(images_dir.glob('*.jpg')))
        
        logger.info(f"\n处理 {split} 集: {len(image_files)} 张图像")
        
        if len(image_files) == 0:
            logger.warning(f"{split} 集没有图像，跳过")
            continue
        
        # 准备任务参数
        manager = Manager()
        counter = manager.Value('i', 0)
        lock = manager.Lock()
        
        tasks = []
        for idx, image_path in enumerate(image_files):
            should_visualize = idx < visualize_samples
            task = (
                image_path,
                split_superpixel_dir,
                split_visualization_dir,
                n_segments,
                compactness,
                sigma,
                should_visualize,
                counter,
                lock
            )
            tasks.append(task)
        
        # 使用进程池并行处理
        logger.info(f"启动 {n_workers} 个工作进程...")
        
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_image, tasks)
        
        # 统计结果
        stats = {
            'total_images': len(image_files),
            'processed': 0,
            'failed': 0,
            'avg_superpixels': 0,
            'failed_images': []
        }
        
        total_superpixels = 0
        for result in results:
            if result['success']:
                stats['processed'] += 1
                total_superpixels += result['num_superpixels']
            else:
                stats['failed'] += 1
                stats['failed_images'].append({
                    'name': result['image_name'],
                    'error': result['error']
                })
        
        # 计算平均超像素数量
        if stats['processed'] > 0:
            stats['avg_superpixels'] = total_superpixels / stats['processed']
        
        # 计算处理时间
        split_duration = (datetime.now() - split_start_time).total_seconds()
        stats['processing_time_seconds'] = split_duration
        stats['images_per_second'] = len(image_files) / split_duration if split_duration > 0 else 0
        
        # 打印统计信息
        logger.info(f"\n{split} 集处理完成:")
        logger.info(f"  成功处理: {stats['processed']} 张")
        logger.info(f"  处理失败: {stats['failed']} 张")
        logger.info(f"  平均超像素数量: {stats['avg_superpixels']:.1f}")
        logger.info(f"  处理时间: {split_duration:.2f} 秒")
        logger.info(f"  处理速度: {stats['images_per_second']:.2f} 张/秒")
        
        if stats['failed'] > 0:
            logger.warning(f"  失败的图像: {[img['name'] for img in stats['failed_images'][:5]]}")
        
        # 保存统计信息
        stats_path = split_superpixel_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 总处理时间
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 超像素分割预处理完成!")
    logger.info(f"总处理时间: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    logger.info(f"\n输出目录结构:")
    logger.info(f"  {output_path}/")
    logger.info(f"    ├── superpixels/")
    logger.info(f"    │   ├── train/")
    logger.info(f"    │   │   ├── *_segments.npy (超像素分割数组)")
    logger.info(f"    │   │   ├── *_info.json (超像素信息)")
    logger.info(f"    │   │   └── stats.json (统计信息)")
    logger.info(f"    │   ├── val/")
    logger.info(f"    │   └── test/")
    logger.info(f"    └── visualizations/")
    logger.info(f"        ├── train/ (前{visualize_samples}张可视化)")
    logger.info(f"        ├── val/")
    logger.info(f"        └── test/")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="超像素分割预处理（多核并行版）")
    parser.add_argument('--data-root', type=str, 
                        default=str(Path(__file__).parent.parent / "data"),
                        help='原始数据根目录（包含 images/<split>）')
    parser.add_argument('--output-root', type=str, 
                        default=str(Path(__file__).parent.parent / "preprocessed"),
                        help='预处理输出根目录（preprocessed）')
    parser.add_argument('--n-segments', type=int, default=500, 
                        help='每张图像的超像素数量')
    parser.add_argument('--compactness', type=float, default=10, 
                        help='超像素紧凑度（值越大越方形）')
    parser.add_argument('--sigma', type=float, default=1, 
                        help='高斯平滑的sigma值')
    parser.add_argument('--visualize-samples', type=int, default=10, 
                        help='每个 split 可视化样本数')
    parser.add_argument('--split', type=str, default=None, 
                        help='单个分割名，如 test_noisy')
    parser.add_argument('--splits', type=str, default=None, 
                        help='多个分割名，逗号分隔，如 "train,val,test_noisy"')
    parser.add_argument('--n-workers', type=int, default=None, 
                        help='工作进程数量（默认为CPU核心数-2）')

    args = parser.parse_args()

    # 解析 splits
    splits = None
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    process_dataset(
        dataset_root=Path(args.data_root),
        output_root=Path(args.output_root),
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        visualize_samples=args.visualize_samples,
        splits=splits,
        n_workers=args.n_workers
    )
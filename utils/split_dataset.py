"""
数据集划分脚本
功能：从训练集中划分出临时验证集（20%），用于模型开发和调优

比赛数据说明：
- 训练集(train): 已公布，用于模型训练
- 测试集(test): 已公布但无标签，可用于中期测试/排行榜
- 验证集(val): 比赛截止前24小时公布，用于最终提交

策略：
1. 现阶段：从训练集划分临时验证集进行模型开发
2. 官方验证集公布后：
   - 恢复训练集(restore模式)
   - 使用全部训练数据重新训练最优模型
   - 在官方验证集上进行最终预测提交
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple

def get_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    获取所有图像和标签文件对
    
    Args:
        images_dir: 图像目录
        labels_dir: 标签目录
    
    Returns:
        [(image_path, label_path), ...] 列表
    """
    pairs = []
    
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(images_dir, img_file)
            
            # 获取对应的标签文件（.txt）
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # 只添加同时存在图像和标签的对
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
            else:
                print(f"警告: 找不到标签文件 {label_file}")
    
    return pairs

def split_dataset(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str,
    split_ratio: float = 0.2,
    seed: int = 42
):
    """
    将训练集划分为训练集和验证集
    
    Args:
        train_images_dir: 原训练集图像目录
        train_labels_dir: 原训练集标签目录
        val_images_dir: 验证集图像目录（将创建）
        val_labels_dir: 验证集标签目录（将创建）
        split_ratio: 验证集比例（默认20%）
        seed: 随机种子
    """
    
    # 设置随机种子以确保可复现
    random.seed(seed)
    
    # 获取所有图像-标签对
    print(f"正在扫描训练集目录...")
    all_pairs = get_image_label_pairs(train_images_dir, train_labels_dir)
    print(f"找到 {len(all_pairs)} 个有效的图像-标签对")
    
    if len(all_pairs) == 0:
        print("错误: 没有找到任何有效的图像-标签对!")
        return
    
    # 随机打乱
    random.shuffle(all_pairs)
    
    # 计算划分点
    val_size = int(len(all_pairs) * split_ratio)
    train_size = len(all_pairs) - val_size
    
    print(f"\n划分方案:")
    print(f"  训练集: {train_size} 个样本 ({(1-split_ratio)*100:.1f}%)")
    print(f"  验证集: {val_size} 个样本 ({split_ratio*100:.1f}%)")
    
    # 划分数据
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]
    
    # 创建验证集目录
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # 移动验证集文件
    print(f"\n正在移动验证集文件...")
    for img_path, label_path in val_pairs:
        # 移动图像
        shutil.move(img_path, os.path.join(val_images_dir, os.path.basename(img_path)))
        # 移动标签
        shutil.move(label_path, os.path.join(val_labels_dir, os.path.basename(label_path)))
    
    print(f"✓ 验证集创建完成!")
    print(f"  验证集图像目录: {val_images_dir}")
    print(f"  验证集标签目录: {val_labels_dir}")
    
    # 验证结果
    train_count = len(os.listdir(train_images_dir))
    val_count = len(os.listdir(val_images_dir))
    
    print(f"\n最终统计:")
    print(f"  训练集图像: {train_count}")
    print(f"  验证集图像: {val_count}")
    print(f"  总计: {train_count + val_count}")

def restore_dataset(
    train_images_dir: str,
    train_labels_dir: str,
    val_images_dir: str,
    val_labels_dir: str
):
    """
    恢复数据集：将验证集的文件移回训练集
    （用于官方验证集公布后，需要使用全部训练数据重训练）
    
    Args:
        train_images_dir: 训练集图像目录
        train_labels_dir: 训练集标签目录
        val_images_dir: 验证集图像目录
        val_labels_dir: 验证集标签目录
    """
    
    if not os.path.exists(val_images_dir):
        print("验证集目录不存在，无需恢复")
        return
    
    print(f"正在将验证集移回训练集...")
    
    moved_count = 0
    
    # 移动图像
    for img_file in os.listdir(val_images_dir):
        src = os.path.join(val_images_dir, img_file)
        dst = os.path.join(train_images_dir, img_file)
        shutil.move(src, dst)
        moved_count += 1
    
    # 移动标签
    for label_file in os.listdir(val_labels_dir):
        src = os.path.join(val_labels_dir, label_file)
        dst = os.path.join(train_labels_dir, label_file)
        shutil.move(src, dst)
    
    # 删除空的验证集目录
    os.rmdir(val_images_dir)
    os.rmdir(val_labels_dir)
    
    print(f"✓ 恢复完成! 移动了 {moved_count} 个图像文件")
    print(f"  训练集图像: {len(os.listdir(train_images_dir))}")

def create_augmented_dataset(
    source_images_dir: str,
    source_labels_dir: str,
    target_images_dir: str,
    target_labels_dir: str,
    augmentation_factor: int = 3
):
    """
    创建增强数据集：对源数据集进行图像增强以扩充数据
    
    Args:
        source_images_dir: 源图像目录
        source_labels_dir: 源标签目录
        target_images_dir: 目标图像目录
        target_labels_dir: 目标标签目录
        augmentation_factor: 增强倍数
    """
    try:
        from preprocessing.image_enhancement import ContainerImageEnhancer
        import cv2
        import numpy as np
        
        print(f"正在创建增强数据集...")
        print(f"源目录: {source_images_dir}")
        print(f"目标目录: {target_images_dir}")
        print(f"增强倍数: {augmentation_factor}")
        
        # 创建目标目录
        os.makedirs(target_images_dir, exist_ok=True)
        os.makedirs(target_labels_dir, exist_ok=True)
        
        # 获取所有图像-标签对
        pairs = get_image_label_pairs(source_images_dir, source_labels_dir)
        print(f"找到 {len(pairs)} 个样本用于增强")
        
        # 创建增强器
        enhancer = ContainerImageEnhancer(img_size=(640, 640))
        
        # 处理每个样本
        for i, (img_path, label_path) in enumerate(pairs):
            print(f"处理样本 {i+1}/{len(pairs)}: {os.path.basename(img_path)}")
            
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"  跳过无效图像: {img_path}")
                continue
            
            # 读取标签
            with open(label_path, 'r') as f:
                labels = f.readlines()
            
            # 转换标签格式
            bboxes = []
            for line in labels:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 转换为 [x_min, y_min, x_max, y_max, class_id] 格式
                    x_min = (x_center - width/2) * image.shape[1]
                    y_min = (y_center - height/2) * image.shape[0]
                    x_max = (x_center + width/2) * image.shape[1]
                    y_max = (y_center + height/2) * image.shape[0]
                    bboxes.append([x_min, y_min, x_max, y_max, class_id])
            
            # 生成增强样本
            for j in range(augmentation_factor):
                try:
                    # 应用增强
                    enhanced_image, enhanced_bboxes = enhancer.enhance_for_detection(
                        image, bboxes, enhance_level='moderate'
                    )
                    
                    # 生成文件名
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    aug_img_name = f"{base_name}_aug_{j+1}.jpg"
                    aug_label_name = f"{base_name}_aug_{j+1}.txt"
                    
                    # 保存增强图像
                    img_np = enhanced_image.numpy().transpose(1, 2, 0)
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(target_images_dir, aug_img_name), 
                               cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    
                    # 转换边界框回YOLO格式并保存
                    with open(os.path.join(target_labels_dir, aug_label_name), 'w') as f:
                        for bbox in enhanced_bboxes:
                            x_min, y_min, x_max, y_max, class_id = bbox
                            # 转换为YOLO格式
                            x_center = ((x_min + x_max) / 2) / 640.0
                            y_center = ((y_min + y_max) / 2) / 640.0
                            width = (x_max - x_min) / 640.0
                            height = (y_max - y_min) / 640.0
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    print(f"  生成增强样本: {aug_img_name}")
                    
                except Exception as e:
                    print(f"  增强样本 {j+1} 时出错: {e}")
                    continue
        
        print(f"✓ 增强数据集创建完成!")
        print(f"  增强图像: {len(os.listdir(target_images_dir))}")
        print(f"  增强标签: {len(os.listdir(target_labels_dir))}")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装必要的依赖包")
    except Exception as e:
        print(f"❌ 创建增强数据集时出错: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集划分工具')
    parser.add_argument('--mode', type=str, choices=['split', 'restore', 'augment'], default='split',
                        help='操作模式: split(划分) 或 restore(恢复) 或 augment(增强)')
    parser.add_argument('--ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2 即 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据根目录 (默认: data)')
    parser.add_argument('--aug-factor', type=int, default=3,
                        help='数据增强倍数 (默认: 3)')
    
    args = parser.parse_args()
    
    # 设置路径
    train_images = os.path.join(args.data_dir, 'images', 'train')
    train_labels = os.path.join(args.data_dir, 'labels', 'train')
    val_images = os.path.join(args.data_dir, 'images', 'val')
    val_labels = os.path.join(args.data_dir, 'labels', 'val')
    aug_images = os.path.join(args.data_dir, 'images', 'augmented')
    aug_labels = os.path.join(args.data_dir, 'labels', 'augmented')
    
    print("=" * 60)
    print("数据集划分工具")
    print("=" * 60)
    
    if args.mode == 'split':
        print(f"\n模式: 划分数据集")
        print(f"验证集比例: {args.ratio * 100:.1f}%")
        print(f"随机种子: {args.seed}")
        
        # 检查是否已存在验证集
        if os.path.exists(val_images) and len(os.listdir(val_images)) > 0:
            print("\n警告: 验证集目录已存在且不为空!")
            response = input("是否要先恢复数据集然后重新划分? (y/n): ")
            if response.lower() == 'y':
                restore_dataset(train_images, train_labels, val_images, val_labels)
            else:
                print("操作已取消")
                exit(0)
        
        split_dataset(
            train_images, train_labels,
            val_images, val_labels,
            split_ratio=args.ratio,
            seed=args.seed
        )
        
    elif args.mode == 'restore':
        print(f"\n模式: 恢复数据集（合并验证集到训练集）")
        response = input("确认要将验证集合并回训练集吗? (y/n): ")
        if response.lower() == 'y':
            restore_dataset(train_images, train_labels, val_images, val_labels)
        else:
            print("操作已取消")
    
    elif args.mode == 'augment':
        print(f"\n模式: 数据增强")
        print(f"增强倍数: {args.aug_factor}")
        response = input("确认要创建增强数据集吗? (y/n): ")
        if response.lower() == 'y':
            create_augmented_dataset(
                train_images, train_labels,
                aug_images, aug_labels,
                augmentation_factor=args.aug_factor
            )
        else:
            print("操作已取消")
    
    print("\n" + "=" * 60)
    print("操作完成!")
    print("=" * 60)
"""
集装箱损伤检测图像增强模块（含离线数据集生成）

新增功能:
1. 离线数据集生成
2. 反归一化并转为 uint8 BGR
3. YOLO 格式标注输出
"""

import os
import shutil

# 临时解决 OpenMP 重复加载问题（不安全，仅用于调试或临时运行）
# 推荐的长期解决方案是确保所有数值/本机扩展使用同一 OpenMP 运行时来源（例如通过 conda 统一安装来自同一 channel 的包）。
if os.name == 'nt':
    # 允许重复加载 libiomp5md.dll（不推荐长期使用）
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable, Union
from enum import Enum
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import logging
import json
from datetime import datetime
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_imread(path: Union[str, Path]) -> Optional[np.ndarray]:
    """安全读取图片，支持包含非 ASCII 字符的 Windows 路径。

    说明：cv2.imread 在某些 OpenCV/Windows 编译下会无法处理含中文或特殊字符的路径，
    使用 numpy.fromfile + cv2.imdecode 可以兼容这类路径。
    """
    p = str(path)
    try:
        # 以二进制方式从文件读取到 numpy 数组，然后解码为图像
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        try:
            # 最后回退到普通 imread（若路径无特殊字符，此分支可命中）
            return cv2.imread(p)
        except Exception:
            return None


class EnhanceLevel(Enum):
    """增强级别枚举"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    CUSTOM = "custom"


@dataclass
class AugmentationConfig:
    """数据增强配置类"""
    
    # 图像尺寸
    img_size: Tuple[int, int] = (640, 640)
    
    # 基础增强参数
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 10
    sat_shift_limit: int = 20
    val_shift_limit: int = 10
    gamma_limit: Tuple[int, int] = (80, 120)
    
    # 几何变换参数
    rotate_limit: Tuple[int, int] = (-10, 10)
    translate_percent: Tuple[float, float] = (0.05, 0.05)
    scale_limit: Tuple[float, float] = (0.9, 1.1)
    shear_limit: Tuple[int, int] = (-5, 5)
    perspective_scale: Tuple[float, float] = (0.05, 0.1)
    
    # 噪声和模糊参数
    noise_intensity: Tuple[float, float] = (0.1, 0.5)
    blur_limit: int = 7
    motion_blur_limit: int = 7
    median_blur_limit: int = 5
    
    # 天气效果参数
    shadow_num_range: Tuple[int, int] = (1, 2)
    rain_drop_length: int = 20
    fog_coef_range: Tuple[float, float] = (0.3, 0.5)
    
    # 损伤特定增强参数
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    grid_distortion_steps: int = 5
    grid_distortion_limit: float = 0.3
    
    # 归一化参数
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # 概率控制
    basic_prob: float = 0.5
    geometric_prob: float = 0.5
    noise_prob: float = 0.3
    blur_prob: float = 0.3
    weather_prob: float = 0.2
    damage_prob: float = 0.2
    
    def validate(self) -> bool:
        """验证配置参数的有效性"""
        try:
            assert self.img_size[0] > 0 and self.img_size[1] > 0, "图像尺寸必须为正数"
            assert 0 <= self.brightness_limit <= 1, "亮度限制必须在[0,1]范围内"
            assert 0 <= self.contrast_limit <= 1, "对比度限制必须在[0,1]范围内"
            assert all(0 <= p <= 1 for p in [self.basic_prob, self.geometric_prob, 
                                              self.noise_prob, self.blur_prob,
                                              self.weather_prob, self.damage_prob]), \
                   "概率值必须在[0,1]范围内"
            return True
        except AssertionError as e:
            logger.error(f"配置验证失败: {e}")
            return False


class TransformCache:
    """Transform对象缓存，避免重复创建"""
    
    def __init__(self):
        self._cache: Dict[str, A.Compose] = {}
    
    def get(self, key: str) -> Optional[A.Compose]:
        """获取缓存的transform"""
        return self._cache.get(key)
    
    def set(self, key: str, transform: A.Compose):
        """设置缓存的transform"""
        self._cache[key] = transform
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()


class ContainerImageEnhancer:
    """集装箱图像增强器（优化版+数据集生成）"""
    
    def __init__(self, 
                 config: Optional[AugmentationConfig] = None,
                 enable_cache: bool = True):
        """
        初始化图像增强器
        
        参数:
            config: 增强配置对象，如果为None则使用默认配置
            enable_cache: 是否启用transform缓存
        """
        self.config = config or AugmentationConfig()
        
        # 验证配置
        if not self.config.validate():
            raise ValueError("配置参数无效")
        
        self.enable_cache = enable_cache
        self._cache = TransformCache() if enable_cache else None
        
        # 初始化所有transform流水线
        self._setup_transforms()
        
        logger.info(f"图像增强器初始化完成 - 尺寸: {self.config.img_size}, 缓存: {enable_cache}")
    def _setup_transforms(self):
        """设置所有数据增强流水线"""
        cfg = self.config
    
        # 基础增强
        self.basic_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                brightness_by_max=True,
                p=cfg.basic_prob
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.sat_shift_limit,
                val_shift_limit=cfg.val_shift_limit,
                p=0.3
            ),
            A.RandomGamma(
                gamma_limit=cfg.gamma_limit,
                p=0.3
            ),
        ])
    
        # 几何增强
        self.geometric_transform = A.Compose([
            A.Affine(
                rotate=cfg.rotate_limit,
                translate_percent=cfg.translate_percent,
                scale=cfg.scale_limit,
                shear=cfg.shear_limit,
                p=cfg.geometric_prob
            ),
            A.Perspective(
                scale=cfg.perspective_scale,
                keep_size=True,
                p=0.3
            ),
        ])
    
        # 噪声增强 - 修复
        self.noise_transform = A.Compose([
            A.OneOf([
                A.ISONoise(
                    intensity=cfg.noise_intensity,
                    color_shift=(0.01, 0.05),
                    p=1.0
                ),
                # GaussNoise 在新版本中使用 std_range / mean_range，std_range 要在 [0,1]
                A.GaussNoise(std_range=cfg.noise_intensity, mean_range=(0.0, 0.0), p=1.0),
            ], p=cfg.noise_prob),
        ])
    
        # 模糊增强
        self.blur_transform = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=cfg.motion_blur_limit, p=1.0),
                A.GaussianBlur(blur_limit=cfg.blur_limit, p=1.0),
                A.MedianBlur(blur_limit=cfg.median_blur_limit, p=1.0),
            ], p=cfg.blur_prob),
        ])
    
        # 天气效果增强 - 修复
        self.weather_transform = A.Compose([
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_limit=(1, 2),  # 修复参数名
                    shadow_dimension=5,
                    p=1.0
                ),
                A.RandomRain(
                    # 新版 RandomRain 使用 slant_range
                    slant_range=(-10, 10),
                    drop_length=cfg.rain_drop_length,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=7,
                    brightness_coefficient=0.7,
                    rain_type='default',
                    p=1.0
                ),
                A.RandomFog(
                    # 使用 fog_coef_range 传递雾强范围
                    fog_coef_range=cfg.fog_coef_range,
                    alpha_coef=0.08,
                    p=1.0
                ),
            ], p=cfg.weather_prob),
        ])
    
        # 损伤特定增强
        self.damage_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(
                    # 新版 ElasticTransform 不使用 alpha_affine 参数
                    alpha=cfg.elastic_alpha,
                    sigma=cfg.elastic_sigma,
                    p=1.0
                ),
                A.GridDistortion(
                    num_steps=cfg.grid_distortion_steps,
                    distort_limit=cfg.grid_distortion_limit,
                    p=1.0
                ),
            ], p=cfg.damage_prob),
        ])
    
        # 最终预处理（用于训练）
        self.final_transform = A.Compose([
            A.Resize(height=cfg.img_size[0], width=cfg.img_size[1]),
            A.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
            A.ToTensorV2(),
        ])
    
        # 数据集生成transform（不包含归一化和ToTensor）
        self.dataset_transform = A.Compose([
            A.Resize(height=cfg.img_size[0], width=cfg.img_size[1]),
        ])
    
    def _get_augmentation_pipeline(self, 
                                   level: Union[EnhanceLevel, str],
                                   for_detection: bool = False,
                                   for_dataset: bool = False) -> A.Compose:
        """
        根据增强级别获取增强流水线
        
        参数:
            level: 增强级别
            for_detection: 是否用于检测任务（需要处理bbox）
            for_dataset: 是否用于数据集生成（不进行归一化）
            
        返回:
            组合的transform对象
        """
        # 转换为枚举类型
        if isinstance(level, str):
            level = EnhanceLevel(level)
        
        # 生成缓存键
        cache_key = f"{level.value}_det_{for_detection}_ds_{for_dataset}"
        
        # 尝试从缓存获取
        if self.enable_cache and self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # 构建transform列表
        transforms_list = []
        
        if level == EnhanceLevel.LIGHT:
            transforms_list = [self.basic_transform]
        elif level == EnhanceLevel.MODERATE:
            transforms_list = [
                self.basic_transform,
                self.geometric_transform,
                self.noise_transform,
            ]
        elif level == EnhanceLevel.HEAVY:
            transforms_list = [
                self.basic_transform,
                self.geometric_transform,
                self.noise_transform,
                self.blur_transform,
                self.weather_transform,
                self.damage_transform,
            ]
        
        # 展平所有transforms
        flattened_transforms = []
        for t in transforms_list:
            if hasattr(t, 'transforms'):
                flattened_transforms.extend(t.transforms)
            else:
                flattened_transforms.append(t)
        
        # 添加resize
        flattened_transforms.append(
            A.Resize(height=self.config.img_size[0], width=self.config.img_size[1])
        )
        
        # 如果不是用于数据集生成，添加归一化和ToTensor
        if not for_dataset:
            flattened_transforms.append(
                A.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)
            )
            flattened_transforms.append(ToTensorV2())
        
        # 创建最终的Compose
        if for_detection:
            combined_transform = A.Compose(
                flattened_transforms,
                bbox_params=A.BboxParams(
                    format='albumentations',
                    label_fields=['class_labels'],
                    min_visibility=0.3,
                )
            )
        else:
            combined_transform = A.Compose(flattened_transforms)
        
        # 缓存transform
        if self.enable_cache and self._cache:
            self._cache.set(cache_key, combined_transform)
        
        return combined_transform
    
    def enhance_image(self, 
                     image: np.ndarray, 
                     bbox: Optional[List] = None,
                     enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE,
                     return_numpy: bool = False) -> Dict:
        """
        增强单张图像
        
        参数:
            image: 输入图像 (BGR格式)
            bbox: 边界框 [x_min, y_min, x_max, y_max] (可选)
            enhance_level: 增强级别
            return_numpy: 是否返回numpy数组而非tensor
            
        返回:
            包含增强后图像和相关信息的字典
        """
        try:
            # 输入验证
            if image is None or image.size == 0:
                raise ValueError("输入图像为空")
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError(f"图像格式错误，期望(H,W,3)，得到{image.shape}")
            
            # 转换为RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 获取transform
            transform = self._get_augmentation_pipeline(enhance_level, for_detection=False)
            
            # 应用增强
            augmented = transform(image=image_rgb)
            processed_image = augmented['image']
            
            # 如果需要返回numpy格式
            if return_numpy:
                if isinstance(processed_image, torch.Tensor):
                    processed_image = processed_image.numpy()
            
            return {
                'image': processed_image,
                'original_shape': image.shape[:2],
                'enhanced_shape': self.config.img_size,
                'bbox': bbox,
                'enhance_level': enhance_level.value if isinstance(enhance_level, EnhanceLevel) else enhance_level
            }
            
        except Exception as e:
            logger.error(f"图像增强失败: {e}")
            raise
    
    def enhance_batch(self,
                     images: List[np.ndarray],
                     bboxes: Optional[List[List]] = None,
                     enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        """
        批量增强图像
        
        参数:
            images: 图像列表
            bboxes: 边界框列表（可选）
            enhance_level: 增强级别
            progress_callback: 进度回调函数 callback(current, total)
            
        返回:
            增强结果列表
        """
        results = []
        total = len(images)
        
        for idx, image in enumerate(images):
            bbox = bboxes[idx] if bboxes and idx < len(bboxes) else None
            
            try:
                result = self.enhance_image(image, bbox, enhance_level)
                results.append(result)
            except Exception as e:
                logger.warning(f"图像 {idx} 增强失败: {e}")
                results.append(None)
            
            # 调用进度回调
            if progress_callback:
                progress_callback(idx + 1, total)
        
        return results
    
    def enhance_for_detection(self,
                            image: np.ndarray,
                            bboxes: List[List[float]],
                            enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        为检测任务增强图像和边界框
        
        参数:
            image: 输入图像 (BGR格式)
            bboxes: 边界框列表 [[x_min, y_min, x_max, y_max, class_id], ...]
            enhance_level: 增强级别
            
        返回:
            (增强后的图像tensor, 变换后的边界框列表)
        """
        try:
            # 输入验证
            if image is None or image.size == 0:
                raise ValueError("输入图像为空")
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 准备边界框数据
            if bboxes and len(bboxes) > 0:
                bboxes_array = np.array([box[:4] for box in bboxes], dtype=np.float32)
                class_ids = [int(box[4]) for box in bboxes]
                
                # 归一化边界框到[0,1]
                bboxes_norm = bboxes_array.copy()
                bboxes_norm[:, [0, 2]] /= width
                bboxes_norm[:, [1, 3]] /= height
                
                # 裁剪到有效范围
                bboxes_norm = np.clip(bboxes_norm, 0.0, 1.0)
                
            else:
                bboxes_norm = []
                class_ids = []
            
            # 获取transform
            transform = self._get_augmentation_pipeline(enhance_level, for_detection=True)
            
            # 应用增强
            if len(bboxes_norm) > 0:
                augmented = transform(
                    image=image_rgb,
                    bboxes=bboxes_norm.tolist(),
                    class_labels=class_ids
                )
            else:
                # 没有边界框时使用普通transform
                transform_no_bbox = self._get_augmentation_pipeline(enhance_level, for_detection=False)
                augmented = transform_no_bbox(image=image_rgb)
            
            enhanced_image = augmented['image']
            
            # 处理输出边界框
            if 'bboxes' in augmented and len(augmented['bboxes']) > 0:
                bboxes_out = np.array(augmented['bboxes'], dtype=np.float32)
                class_ids_out = augmented['class_labels']
                
                # 还原到绝对坐标
                h_out, w_out = self.config.img_size
                bboxes_out[:, [0, 2]] *= w_out
                bboxes_out[:, [1, 3]] *= h_out
                
                # 组合边界框和类别ID
                bboxes_final = [
                    [*bbox.tolist(), class_id]
                    for bbox, class_id in zip(bboxes_out, class_ids_out)
                ]
            else:
                bboxes_final = []
            
            return enhanced_image, bboxes_final
            
        except Exception as e:
            logger.error(f"检测任务图像增强失败: {e}")
            raise
    
    def enhance_with_mosaic(self,
                           images: List[np.ndarray],
                           bboxes: Optional[List[List[List[float]]]] = None) -> Dict:
        """
        Mosaic增强：将4张图像拼接成一张
        
        参数:
            images: 4张图像列表
            bboxes: 对应的边界框列表 (可选)
            
        返回:
            拼接后的图像和边界框
        """
        if len(images) != 4:
            raise ValueError(f"Mosaic增强需要恰好4张图像，当前提供了{len(images)}张")
        
        try:
            target_h, target_w = self.config.img_size
            
            # 创建画布
            mosaic_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
            mosaic_bboxes = []
            
            # 随机选择分割点
            xc = int(random.uniform(target_w * 0.25, target_w * 0.75))
            yc = int(random.uniform(target_h * 0.25, target_h * 0.75))
            
            # 处理每张图像
            for i, img in enumerate(images):
                # 转换为RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 轻度增强
                augmented = self.basic_transform(image=img_rgb)
                img_rgb = augmented['image']
                
                h, w = img_rgb.shape[:2]
                
                # 计算放置位置
                if i == 0:  # 左上
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # 右上
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_w), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # 左下
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_h, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                else:  # 右下
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_w), min(target_h, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                
                # 放置图像
                mosaic_img[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]
                
                # 转换边界框（如果有）
                if bboxes and i < len(bboxes) and bboxes[i]:
                    for bbox in bboxes[i]:
                        # bbox格式: [x_min, y_min, x_max, y_max, class_id]
                        x_min, y_min, x_max, y_max = bbox[:4]
                        class_id = bbox[4] if len(bbox) > 4 else 0
                        
                        # 调整边界框坐标
                        x_min_new = x1a + (x_min - x1b) * (x2a - x1a) / (x2b - x1b)
                        y_min_new = y1a + (y_min - y1b) * (y2a - y1a) / (y2b - y1b)
                        x_max_new = x1a + (x_max - x1b) * (x2a - x1a) / (x2b - x1b)
                        y_max_new = y1a + (y_max - y1b) * (y2a - y1a) / (y2b - y1b)
                        
                        # 裁剪到画布范围内
                        x_min_new = max(0, min(x_min_new, target_w))
                        y_min_new = max(0, min(y_min_new, target_h))
                        x_max_new = max(0, min(x_max_new, target_w))
                        y_max_new = max(0, min(y_max_new, target_h))
                        
                        # 检查边界框是否有效
                        if x_max_new > x_min_new and y_max_new > y_min_new:
                            mosaic_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new, class_id])
            
            # 最终预处理
            final_augmented = self.final_transform(image=mosaic_img)
            processed_image = final_augmented['image']
            
            return {
                'image': processed_image,
                'bboxes': mosaic_bboxes,
                'mosaic': True,
                'split_point': (xc, yc)
            }
            
        except Exception as e:
            logger.error(f"Mosaic增强失败: {e}")
            raise
    
    def enhance_for_classification(self,
                                  image: np.ndarray,
                                  enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE) -> torch.Tensor:
        """
        为分类任务增强图像
        
        参数:
            image: 输入图像 (BGR格式)
            enhance_level: 增强级别
            
        返回:
            增强后的PyTorch张量
        """
        result = self.enhance_image(image, enhance_level=enhance_level, return_numpy=False)
        return result['image']
    
    def denormalize_image(self, 
                         normalized_image: Union[torch.Tensor, np.ndarray],
                         to_bgr: bool = True,
                         to_uint8: bool = True) -> np.ndarray:
        """
        反归一化图像
        
        参数:
            normalized_image: 归一化后的图像 (Tensor或numpy数组)
            to_bgr: 是否转换为BGR格式
            to_uint8: 是否转换为uint8类型
            
        返回:
            反归一化后的图像 (numpy数组)
        """
        # 转换为numpy数组
        if isinstance(normalized_image, torch.Tensor):
            img = normalized_image.cpu().numpy()
        else:
            img = normalized_image.copy()
        
        # 如果是CHW格式，转换为HWC
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        # 反归一化
        mean = np.array(self.config.normalize_mean).reshape(1, 1, 3)
        std = np.array(self.config.normalize_std).reshape(1, 1, 3)
        img = img * std + mean
        
        # 裁剪到[0, 1]范围
        img = np.clip(img, 0, 1)
        
        # 转换为uint8
        if to_uint8:
            img = (img * 255).astype(np.uint8)
        
        # 转换为BGR
        if to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def xyxy_to_yolo(self, 
                    bbox: List[float], 
                    img_width: int, 
                    img_height: int) -> Tuple[float, float, float, float]:
        """
        将xyxy格式的边界框转换为YOLO格式
        
        参数:
            bbox: [x_min, y_min, x_max, y_max] 格式的边界框
            img_width: 图像宽度
            img_height: 图像高度
            
        返回:
            (x_center, y_center, width, height) 归一化到[0,1]的YOLO格式
        """
        x_min, y_min, x_max, y_max = bbox[:4]
        
        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # 归一化
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # 裁剪到[0, 1]范围
        x_center = np.clip(x_center, 0, 1)
        y_center = np.clip(y_center, 0, 1)
        width = np.clip(width, 0, 1)
        height = np.clip(height, 0, 1)
        
        return x_center, y_center, width, height
    
    def save_yolo_annotation(self, 
                            bboxes: List[List[float]], 
                            save_path: Union[str, Path],
                            img_width: int,
                            img_height: int):
        """
        保存YOLO格式的标注文件
        
        参数:
            bboxes: 边界框列表 [[x_min, y_min, x_max, y_max, class_id], ...]
            save_path: 保存路径
            img_width: 图像宽度
            img_height: 图像高度
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            for bbox in bboxes:
                class_id = int(bbox[4])
                x_center, y_center, width, height = self.xyxy_to_yolo(
                    bbox[:4], img_width, img_height
                )
                # YOLO格式: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def generate_offline_dataset(self,
                                images: List[np.ndarray],
                                bboxes: List[List[List[float]]],
                                output_dir: Union[str, Path],
                                image_names: Optional[List[str]] = None,
                                enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE,
                                split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                                augment_per_image: int = 1,
                                class_names: Optional[List[str]] = None) -> Dict:
        """
        生成离线数据集（YOLO格式）
        
        参数:
            images: 图像列表 (BGR格式)
            bboxes: 边界框列表 [image_idx][bbox_idx][x_min, y_min, x_max, y_max, class_id]
            output_dir: 输出目录
            image_names: 图像名称列表（不含扩展名）
            enhance_level: 增强级别
            split_ratio: 数据集划分比例 (train, val, test)
            augment_per_image: 每张图像生成的增强样本数
            class_names: 类别名称列表
            
        返回:
            数据集统计信息
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录，使用 data/images/<split> 和 data/labels/<split> 结构，便于后续一键预处理使用
        for split in ['train', 'val', 'test']:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # 生成图像名称：如果没有提供，则使用 image_000000 风格；
        # 如果提供了相对路径（如 train/xxx），只保留文件名部分，避免保存时产生嵌套子目录
        if image_names is None:
            image_names = [f"image_{i:06d}" for i in range(len(images))]
        else:
            # 规范化提供的名称，保持唯一性但移除路径分隔符
            norm_names = []
            for n in image_names:
                try:
                    p = Path(n)
                    name_only = p.with_suffix('').name
                except Exception:
                    name_only = str(n).replace('/', '_').replace('\\', '_')
                norm_names.append(name_only)
            image_names = norm_names
        
        # 数据集划分
        total_images = len(images)
        indices = list(range(total_images))
        random.shuffle(indices)

        # 说明：使用 int() 会向下取整，可能会产生余数。
        # 原实现用两个 int() 切片，剩余的全部落到 test，这会导致 val/test 出现 412/414 这样的不对称。
        # 这里显式计算三个 split 的样本数：先按比例取整 train 和 val，剩余样本分配给 test，保证三者之和等于 total_images。
        train_count = int(total_images * split_ratio[0])
        val_count = int(total_images * split_ratio[1])
        test_count = total_images - train_count - val_count

        # 切片索引
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]

        # 记录分配信息（便于理解为何出现不均等）
        logger.info(f"数据集切分细节: total={total_images}, train={train_count}, val={val_count}, test={test_count}")
        
        splits_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }
        
        stats = {
            'total_original': total_images,
            'total_generated': 0,
            'train': 0,
            'val': 0,
            'test': 0,
            'failed': 0
        }
        
        logger.info(f"开始生成数据集到: {output_dir}")
        logger.info(f"数据集划分 - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        # 处理每个数据集分割
        for split_name, split_indices in splits_indices.items():
            logger.info(f"\n处理 {split_name} 集...")
            
            with tqdm(total=len(split_indices) * augment_per_image, desc=f"{split_name}集") as pbar:
                for idx in split_indices:
                    image = images[idx]
                    image_bboxes = bboxes[idx] if idx < len(bboxes) else []
                    base_name = image_names[idx]
                    
                    # 对每张图像生成多个增强样本
                    for aug_idx in range(augment_per_image):
                        try:
                            # 生成增强图像和边界框
                            enhanced_img, enhanced_bboxes = self._generate_dataset_sample(
                                image, image_bboxes, enhance_level
                            )
                            
                            # 生成文件名
                            if augment_per_image > 1:
                                file_name = f"{base_name}_aug{aug_idx}"
                            else:
                                file_name = base_name
                            
                            # 保存图像到 data/images/<split>/
                            img_path = output_dir / 'images' / split_name / f"{file_name}.jpg"
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(img_path), enhanced_img)
                            
                            # 保存标注到 data/labels/<split>/
                            label_path = output_dir / 'labels' / split_name / f"{file_name}.txt"
                            label_path.parent.mkdir(parents=True, exist_ok=True)
                            self.save_yolo_annotation(
                                enhanced_bboxes,
                                label_path,
                                enhanced_img.shape[1],
                                enhanced_img.shape[0]
                            )
                            
                            stats[split_name] += 1
                            stats['total_generated'] += 1
                            
                        except Exception as e:
                            logger.warning(f"处理 {base_name}_aug{aug_idx} 失败: {e}")
                            stats['failed'] += 1
                        
                        pbar.update(1)
        
        # 保存数据集配置文件（YOLO格式）
        self._save_dataset_yaml(output_dir, class_names, stats)
        
        # 保存统计信息
        self._save_statistics(output_dir, stats)
        
        logger.info("\n" + "="*80)
        logger.info("数据集生成完成!")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"总计生成: {stats['total_generated']} 张图像")
        logger.info(f"Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info("="*80)
        
        return stats
    
    def generate_offline_dataset_v2(self,
                                     images: List[np.ndarray],
                                     bboxes: List[List[List[float]]],
                                     output_dir: Union[str, Path],
                                     image_names: Optional[List[str]] = None,
                                     enhance_level: Union[EnhanceLevel, str] = EnhanceLevel.MODERATE,
                                     split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                                     augment_per_image: int = 1,
                                     class_names: Optional[List[str]] = None,
                                     respect_original_split: bool = True,
                                     preserve_subdirs: bool = True,
                                     random_seed: Optional[int] = None,
                                     clean_output_dir: bool = False,
                                     subdir_timestamp: bool = False) -> Dict:
        """
        生成离线数据集（改进版）：
        - 可保留原始 train/val/test 划分（从 image_names 首段推断）
        - 可按需保留子目录，避免同名覆盖
        - 可设置随机种子，确保划分复现
        - 可清空输出目录或写入时间戳子目录
        """

        # 1) 随机种子（可复现）
        if random_seed is not None:
            try:
                random.seed(random_seed)
                np.random.seed(random_seed)
                try:
                    torch.manual_seed(random_seed)
                except Exception:
                    pass
                logger.info(f"使用随机种子: {random_seed}")
            except Exception as e:
                logger.warning(f"设置随机种子失败: {e}")

        # 2) 输出目录处理（清空/时间戳）
        root_output_dir = Path(output_dir)
        if clean_output_dir and root_output_dir.exists():
            logger.warning(f"将清空输出目录: {root_output_dir}")
            try:
                shutil.rmtree(root_output_dir)
            except Exception as e:
                logger.warning(f"清空输出目录失败: {e}")

        if subdir_timestamp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = root_output_dir / ts
        else:
            output_dir = root_output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # 3) 名称处理（可保留子目录，推断原始划分）
        if image_names is None:
            image_names = [f"image_{i:06d}" for i in range(len(images))]

        processed_names: List[str] = []
        name_splits: List[Optional[str]] = []
        for n in image_names:
            try:
                p = Path(n)
                as_posix_no_ext = p.with_suffix('').as_posix()
            except Exception:
                as_posix_no_ext = str(n).replace('\\', '/').split('.')[0]

            parts = [seg for seg in as_posix_no_ext.split('/') if seg not in ('', '.')]
            split_hint = parts[0] if parts else None
            split_hint = split_hint if split_hint in {'train', 'val', 'test'} else None

            if preserve_subdirs:
                if split_hint and len(parts) > 1:
                    rel_parts = parts[1:]
                elif len(parts) > 0:
                    rel_parts = parts
                else:
                    rel_parts = [as_posix_no_ext]
                rel_name = '/'.join(rel_parts)
            else:
                rel_name = parts[-1] if parts else as_posix_no_ext

            processed_names.append(rel_name)
            name_splits.append(split_hint)

        # 4) 划分（保留原始划分优先）
        total_images = len(images)
        indices = list(range(total_images))

        train_indices: List[int] = []
        val_indices: List[int] = []
        test_indices: List[int] = []
        unspecified_indices: List[int] = []

        if respect_original_split:
            for i, split_hint in enumerate(name_splits):
                if split_hint == 'train':
                    train_indices.append(i)
                elif split_hint == 'val':
                    val_indices.append(i)
                elif split_hint == 'test':
                    test_indices.append(i)
                else:
                    unspecified_indices.append(i)

            if unspecified_indices:
                tmp = unspecified_indices.copy()
                random.shuffle(tmp)
                u_total = len(tmp)
                u_train = int(u_total * split_ratio[0])
                u_val = int(u_total * split_ratio[1])
                u_test = u_total - u_train - u_val
                train_indices.extend(tmp[:u_train])
                val_indices.extend(tmp[u_train:u_train + u_val])
                test_indices.extend(tmp[u_train + u_val:])

            logger.info(
                f"保留原始划分: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}, 未指明={len(unspecified_indices)}"
            )
        else:
            random.shuffle(indices)
            train_count = int(total_images * split_ratio[0])
            val_count = int(total_images * split_ratio[1])
            test_count = total_images - train_count - val_count
            train_indices = indices[:train_count]
            val_indices = indices[train_count:train_count + val_count]
            test_indices = indices[train_count + val_count:]
            logger.info(
                f"重新划分: total={total_images}, train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
            )

        splits_indices = {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        }

        stats = {
            'total_original': total_images,
            'total_generated': 0,
            'train': 0,
            'val': 0,
            'test': 0,
            'failed': 0
        }

        logger.info(f"开始生成数据集于 {output_dir}")
        logger.info(f"数据集划分 - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        # 5) 逐分割生成
        for split_name, split_indices in splits_indices.items():
            logger.info(f"\n处理 {split_name} 集...")
            with tqdm(total=len(split_indices) * augment_per_image, desc=f"{split_name}集") as pbar:
                for idx in split_indices:
                    image = images[idx]
                    image_bboxes = bboxes[idx] if idx < len(bboxes) else []
                    base_name = processed_names[idx]

                    for aug_idx in range(augment_per_image):
                        try:
                            enhanced_img, enhanced_bboxes = self._generate_dataset_sample(
                                image, image_bboxes, enhance_level
                            )

                            if augment_per_image > 1:
                                file_name = f"{base_name}_aug{aug_idx}"
                            else:
                                file_name = base_name

                            img_path = output_dir / 'images' / split_name / f"{file_name}.jpg"
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(img_path), enhanced_img)

                            label_path = output_dir / 'labels' / split_name / f"{file_name}.txt"
                            label_path.parent.mkdir(parents=True, exist_ok=True)
                            self.save_yolo_annotation(
                                enhanced_bboxes,
                                label_path,
                                enhanced_img.shape[1],
                                enhanced_img.shape[0]
                            )

                            stats[split_name] += 1
                            stats['total_generated'] += 1
                        except Exception as e:
                            logger.warning(f"处理 {base_name}_aug{aug_idx} 失败: {e}")
                            stats['failed'] += 1
                        pbar.update(1)

        # 6) 保存配置与统计
        self._save_dataset_yaml(output_dir, class_names, stats)
        self._save_statistics(output_dir, stats)

        logger.info("\n" + "="*80)
        logger.info("数据集生成完成（v2）")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"总计生成: {stats['total_generated']} 张图像")
        logger.info(f"Train: {stats['train']}, Val: {stats['val']}, Test: {stats['test']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info("="*80)

        return stats
    
    def _generate_dataset_sample(self,
                                 image: np.ndarray,
                                 bboxes: List[List[float]],
                                 enhance_level: Union[EnhanceLevel, str]) -> Tuple[np.ndarray, List[List[float]]]:
        """
        生成单个数据集样本（不进行归一化，返回uint8 BGR图像）
        
        参数:
            image: 输入图像 (BGR格式)
            bboxes: 边界框列表
            enhance_level: 增强级别
            
        返回:
            (增强后的BGR图像, 变换后的边界框)
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        
        # 准备边界框数据
        if bboxes and len(bboxes) > 0:
            bboxes_array = np.array([box[:4] for box in bboxes], dtype=np.float32)
            class_ids = [int(box[4]) for box in bboxes]
            
            # 归一化边界框到[0,1]
            bboxes_norm = bboxes_array.copy()
            bboxes_norm[:, [0, 2]] /= width
            bboxes_norm[:, [1, 3]] /= height
            bboxes_norm = np.clip(bboxes_norm, 0.0, 1.0)
        else:
            bboxes_norm = []
            class_ids = []
        
        # 获取数据集生成专用的transform（不包含归一化）
        transform = self._get_augmentation_pipeline(
            enhance_level, 
            for_detection=len(bboxes_norm) > 0,
            for_dataset=True
        )
        
        # 应用增强
        if len(bboxes_norm) > 0:
            augmented = transform(
                image=image_rgb,
                bboxes=bboxes_norm.tolist(),
                class_labels=class_ids
            )
        else:
            augmented = transform(image=image_rgb)
        
        # 获取增强后的图像（RGB格式，uint8）
        enhanced_img_rgb = augmented['image']
        
        # 转换为BGR
        enhanced_img_bgr = cv2.cvtColor(enhanced_img_rgb, cv2.COLOR_RGB2BGR)
        
        # 处理边界框
        if 'bboxes' in augmented and len(augmented['bboxes']) > 0:
            bboxes_out = np.array(augmented['bboxes'], dtype=np.float32)
            class_ids_out = augmented['class_labels']
            
            # 还原到绝对坐标
            h_out, w_out = self.config.img_size
            bboxes_out[:, [0, 2]] *= w_out
            bboxes_out[:, [1, 3]] *= h_out
            
            # 组合边界框和类别ID
            bboxes_final = [
                [*bbox.tolist(), class_id]
                for bbox, class_id in zip(bboxes_out, class_ids_out)
            ]
        else:
            bboxes_final = []
        
        return enhanced_img_bgr, bboxes_final
    
    def _save_dataset_yaml(self, 
                          output_dir: Path, 
                          class_names: Optional[List[str]],
                          stats: Dict):
        """
        保存YOLO格式的数据集配置文件
        
        参数:
            output_dir: 输出目录
            class_names: 类别名称列表
            stats: 统计信息
        """
        yaml_path = output_dir / 'dataset.yaml'
        
        # 默认类别名称
        if class_names is None:
            class_names = [f"class_{i}" for i in range(10)]
        
        yaml_content = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names,
            'stats': stats
        }
        
        # 保存YAML文件
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"数据集配置已保存到: {yaml_path}")
    
    def _save_statistics(self, output_dir: Path, stats: Dict):
        """
        保存统计信息
        
        参数:
            output_dir: 输出目录
            stats: 统计信息
        """
        stats_path = output_dir / 'statistics.json'
        
        stats_with_time = {
            'generation_time': datetime.now().isoformat(),
            'image_size': self.config.img_size,
            'statistics': stats
        }
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_with_time, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计信息已保存到: {stats_path}")
    
    def update_config(self, **kwargs):
        """
        更新配置参数
        
        参数:
            **kwargs: 要更新的配置项
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"更新配置: {key} = {value}")
            else:
                logger.warning(f"未知的配置项: {key}")
        
        # 验证新配置
        if not self.config.validate():
            raise ValueError("更新后的配置参数无效")
        
        # 清空缓存并重新设置transforms
        if self._cache:
            self._cache.clear()
        self._setup_transforms()
    
    def get_config(self) -> AugmentationConfig:
        """获取当前配置"""
        return self.config
    
    def clear_cache(self):
        """清空transform缓存"""
        if self._cache:
            self._cache.clear()
            logger.info("Transform缓存已清空")


def create_enhancer_preset(preset: str = 'default') -> ContainerImageEnhancer:
    """
    创建预设配置的增强器
    
    参数:
        preset: 预设名称 ('default', 'aggressive', 'conservative', 'fast')
        
    返回:
        配置好的增强器实例
    """
    if preset == 'aggressive':
        # 激进模式：更强的增强
        config = AugmentationConfig(
            brightness_limit=0.3,
            contrast_limit=0.3,
            rotate_limit=(-20, 20),
            scale_limit=(0.8, 1.2),
            basic_prob=0.7,
            geometric_prob=0.7,
            noise_prob=0.5,
            weather_prob=0.3,
        )
    elif preset == 'conservative':
        # 保守模式：较弱的增强
        config = AugmentationConfig(
            brightness_limit=0.1,
            contrast_limit=0.1,
            rotate_limit=(-5, 5),
            scale_limit=(0.95, 1.05),
            basic_prob=0.3,
            geometric_prob=0.3,
            noise_prob=0.1,
            weather_prob=0.1,
        )
    elif preset == 'fast':
        # 快速模式：减少增强操作
        config = AugmentationConfig(
            basic_prob=0.4,
            geometric_prob=0.3,
            noise_prob=0.0,
            blur_prob=0.0,
            weather_prob=0.0,
            damage_prob=0.0,
        )
    else:  # default
        config = AugmentationConfig()
    
    return ContainerImageEnhancer(config, enable_cache=True)


def demo_dataset_generation():
    """演示离线数据集生成功能"""
    print("=" * 80)
    print("基于现有数据集生成离线增强数据集：将处理 `数据集3713` 下的真实图像和 YOLO 标签")
    print("=" * 80)

    # 创建增强器
    enhancer = ContainerImageEnhancer()

    # 配置：数据集根目录（相对于项目根或绝对路径）
    dataset_root = Path.cwd() / '数据集3713'
    images_root = dataset_root / 'images'
    labels_root = dataset_root / 'labels'

    if not images_root.exists():
        raise FileNotFoundError(f"找不到图像目录: {images_root}，请确保数据集位于该路径或修改路径")

    print("\n1. 读取数据集文件（递归 images/，读取对应 labels/）...")

    images = []
    bboxes = []
    image_names = []

    # 支持常见图像格式
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for img_path in sorted(images_root.rglob('*')):
        if not img_path.is_file() or img_path.suffix.lower() not in img_exts:
            continue
        img = _safe_imread(img_path)
        if img is None:
            logger.warning(f"无法读取图像: {img_path}, 跳过")
            continue

        # 计算相对路径以寻找对应 label（例如 images/train/img.jpg -> labels/train/img.txt）
        try:
            rel = img_path.relative_to(images_root)
        except Exception:
            rel = img_path.name

        label_path = labels_root / rel.with_suffix('.txt')

        boxes = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    x_c, y_c, w_rel, h_rel = map(float, parts[1:5])
                    h, w = img.shape[:2]
                    x_c *= w
                    y_c *= h
                    bbox_w = w_rel * w
                    bbox_h = h_rel * h
                    x_min = x_c - bbox_w / 2.0
                    y_min = y_c - bbox_h / 2.0
                    x_max = x_c + bbox_w / 2.0
                    y_max = y_c + bbox_h / 2.0
                    boxes.append([x_min, y_min, x_max, y_max, cls])

        images.append(img)
        bboxes.append(boxes)
        # 使用相对路径（无后缀）作为图像名，确保不同子目录下同名文件不会冲突
        image_names.append(rel.with_suffix('').as_posix())

    total_images = len(images)
    print(f"   ✓ 读取到 {total_images} 张图像")

    # 读取类别名称（如果存在 classes.txt）
    classes_file = dataset_root / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = None

    # 生成数据集
    print("\n2. 生成离线数据集...")
    output_dir = Path.cwd() / 'data'

    # 每张图像生成的增强样本数（可按需修改）
    augment_per_image = 1

    stats = enhancer.generate_offline_dataset_v2(
        images=images,
        bboxes=bboxes,
        output_dir=output_dir,
        image_names=image_names,
        enhance_level=EnhanceLevel.MODERATE,
        split_ratio=(0.8, 0.1, 0.1),
        augment_per_image=augment_per_image,
        class_names=class_names,
        respect_original_split=True,
        preserve_subdirs=True,
        random_seed=42,
        clean_output_dir=False,
        subdir_timestamp=True
    )

    # 显示结果
    print("\n3. 生成结果:")
    print(f"   ✓ 输出目录: {output_dir}")
    print(f"   ✓ 原始图像: {stats.get('total_original', total_images)} 张")
    print(f"   ✓ 训练集: {stats['train']} 张")
    print(f"   ✓ 验证集: {stats['val']} 张")
    print(f"   ✓ 测试集: {stats['test']} 张")
    print(f"   ✓ 总计: {stats['total_generated']} 张")

    print("\n4. 数据集目录结构:")
    print("   data/")
    print("   ├── images/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   │   └── test/")
    print("   ├── labels/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   │   └── test/")
    print("   ├── data.yaml")
    print("   └── statistics.json")

    print("\n5. 使用方法:")
    print("   # 训练YOLO模型")
    print("   yolo train data=generated_dataset/dataset.yaml model=yolov8n.pt epochs=100")

    print("=" * 80)


def demo_enhancement():
    """演示图像增强功能"""
    print("=" * 80)
    print("集装箱损伤检测图像增强演示（含数据集生成）")
    print("=" * 80)
    
    # 1. 使用默认配置
    print("\n1. 创建默认增强器")
    enhancer = ContainerImageEnhancer()
    print(f"   ✓ 输出尺寸: {enhancer.config.img_size}")
    print(f"   ✓ 缓存启用: {enhancer.enable_cache}")
    
    # 2. 支持的增强级别
    print("\n2. 支持的增强级别:")
    for level in EnhanceLevel:
        print(f"   - {level.value}")
    
    # 3. 新增功能
    print("\n3. 新增功能:")
    print("   ✓ 离线数据集生成")
    print("   ✓ 反归一化转uint8 BGR")
    print("   ✓ YOLO格式标注输出")
    print("   ✓ 自动数据集划分")
    print("   ✓ 批量增强生成")
    
    # 4. 使用方法
    print("\n" + "=" * 80)
    print("使用方法:")
    print("-" * 80)
    print("# 生成离线数据集")
    print("from image_enhancement_with_dataset_generation import ContainerImageEnhancer")
    print()
    print("enhancer = ContainerImageEnhancer()")
    print()
    print("# 生成数据集（YOLO格式）")
    print("stats = enhancer.generate_offline_dataset(")
    print("    images=images,  # BGR格式图像列表")
    print("    bboxes=bboxes,  # [[x_min, y_min, x_max, y_max, class_id], ...]")
    print("    output_dir='./dataset',")
    print("    enhance_level='moderate',")
    print("    split_ratio=(0.8, 0.1, 0.1),")
    print("    augment_per_image=5,  # 每张图生成5个增强版本")
    print("    class_names=['rust', 'dent', 'hole']")
    print(")")
    print()
    print("# 反归一化图像")
    print("bgr_img = enhancer.denormalize_image(tensor_img, to_bgr=True, to_uint8=True)")
    print()
    print("# 转换为YOLO格式")
    print("x_c, y_c, w, h = enhancer.xyxy_to_yolo(bbox, img_w, img_h)")
    print("=" * 80)


if __name__ == "__main__":
    # demo_enhancement()
    demo_dataset_generation()

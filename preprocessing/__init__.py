"""
预处理模块
包含从原始图像到图数据集的完整预处理流程
"""

from .create_dataset import ContainerGraphDataset

__all__ = ['ContainerGraphDataset']

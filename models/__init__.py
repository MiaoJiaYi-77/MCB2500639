"""
模型定义模块
包含GCN+Transformer和分割模型
"""

from .models import GCNTransformerClassifier, GCNSegmentation

__all__ = ['GCNTransformerClassifier', 'GCNSegmentation']

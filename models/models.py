"""
GCN模型定义
包含问题1(分类)和问题2(分割)的模型架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNTransformerClassifier(nn.Module):
    """
    问题1: GCN + Transformer 图分类模型
    用于判断集装箱图像是否有残损
    """
    
    def __init__(
        self,
        node_features=2048,
        hidden_dim=256,
        gcn_layers=3,
        num_heads=8,
        num_transformer_layers=2,
        num_classes=2,
        dropout=0.3
    ):
        """
        初始化模型
        
        参数:
            node_features: 输入节点特征维度 (ResNet-50: 2048)
            hidden_dim: GCN隐藏层维度
            gcn_layers: GCN层数
            num_heads: Transformer注意力头数
            num_transformer_layers: Transformer编码器层数
            num_classes: 分类类别数 (2: 有残损/无残损)
            dropout: Dropout比率
        """
        super(GCNTransformerClassifier, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.gcn_layers = gcn_layers
        
        # 输入投影层
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # GCN卷积层
        self.gcn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # CLS token (用于图级表示)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyG Data对象
                - x: 节点特征 (N, node_features)
                - edge_index: 边索引 (2, E)
                - batch: 批次索引 (N,)
        
        返回:
            logits: 分类logits (batch_size, num_classes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. 输入投影
        x = self.input_proj(x)  # (N, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 2. GCN卷积层 (局部邻域信息聚合)
        for i, (conv, bn) in enumerate(zip(self.gcn_convs, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            # 残差连接
            if i > 0:
                x = x + x_residual
        
        # 3. 转换为Transformer输入格式
        # 将图中的节点组织成序列
        batch_size = batch.max().item() + 1
        node_sequences = []
        
        for i in range(batch_size):
            # 获取第i个图的所有节点
            mask = (batch == i)
            graph_nodes = x[mask]  # (num_nodes_i, hidden_dim)
            node_sequences.append(graph_nodes)
        
        # 4. Transformer处理 (全局关系建模)
        graph_features = []
        
        for nodes in node_sequences:
            # 添加CLS token
            cls = self.cls_token.expand(1, -1, -1)  # (1, 1, hidden_dim)
            nodes = nodes.unsqueeze(0)  # (1, num_nodes, hidden_dim)
            seq = torch.cat([cls, nodes], dim=1)  # (1, num_nodes+1, hidden_dim)
            
            # Transformer编码
            seq = self.transformer(seq)  # (1, num_nodes+1, hidden_dim)
            
            # 提取CLS token作为图表示
            graph_feat = seq[:, 0, :]  # (1, hidden_dim)
            graph_features.append(graph_feat)
        
        # 拼接所有图的特征
        graph_features = torch.cat(graph_features, dim=0)  # (batch_size, hidden_dim)
        
        # 5. 分类
        logits = self.classifier(graph_features)  # (batch_size, num_classes)
        
        return logits


class GCNSegmentation(nn.Module):
    """
    问题2: GCN节点分类模型
    用于分割不同类型的残损区域
    """
    
    def __init__(
        self,
        node_features=2048,
        hidden_dim=256,
        gcn_layers=3,
        num_classes=10,
        dropout=0.3
    ):
        """
        初始化模型
        
        参数:
            node_features: 输入节点特征维度
            hidden_dim: GCN隐藏层维度
            gcn_layers: GCN层数
            num_classes: 残损类别数 (包括背景)
            dropout: Dropout比率
        """
        super(GCNSegmentation, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.gcn_layers = gcn_layers
        
        # 输入投影层
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        # GCN卷积层
        self.gcn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 节点分类头 (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播
        
        参数:
            data: PyG Data对象
                - x: 节点特征 (N, node_features)
                - edge_index: 边索引 (2, E)
        
        返回:
            logits: 节点分类logits (N, num_classes)
        """
        x, edge_index = data.x, data.edge_index
        
        # 1. 输入投影
        x = self.input_proj(x)  # (N, hidden_dim)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 2. GCN卷积层 (上下文信息聚合)
        for i, (conv, bn) in enumerate(zip(self.gcn_convs, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            # 残差连接
            if i > 0:
                x = x + x_residual
        
        # 3. 节点分类
        logits = self.classifier(x)  # (N, num_classes)
        
        return logits


class HybridGCNModel(nn.Module):
    """
    混合模型: 同时支持分类和分割任务
    共享GCN主干网络
    """
    
    def __init__(
        self,
        node_features=2048,
        hidden_dim=256,
        gcn_layers=3,
        num_heads=8,
        num_transformer_layers=2,
        num_graph_classes=2,
        num_node_classes=10,
        dropout=0.3
    ):
        """
        初始化混合模型
        
        参数:
            node_features: 输入节点特征维度
            hidden_dim: GCN隐藏层维度
            gcn_layers: GCN层数
            num_heads: Transformer注意力头数
            num_transformer_layers: Transformer编码器层数
            num_graph_classes: 图分类类别数
            num_node_classes: 节点分类类别数
            dropout: Dropout比率
        """
        super(HybridGCNModel, self).__init__()
        
        # 共享的GCN主干
        self.input_proj = nn.Linear(node_features, hidden_dim)
        
        self.gcn_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 图分类分支 (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_graph_classes)
        )
        
        # 节点分类分支 (MLP)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_node_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data, task='both'):
        """
        前向传播
        
        参数:
            data: PyG Data对象
            task: 任务类型 ('graph', 'node', 'both')
        
        返回:
            根据task返回不同的输出
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 共享GCN主干
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for i, (conv, bn) in enumerate(zip(self.gcn_convs, self.batch_norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            if i > 0:
                x = x + x_residual
        
        outputs = {}
        
        # 图分类任务
        if task in ['graph', 'both']:
            batch_size = batch.max().item() + 1
            graph_features = []
            
            for i in range(batch_size):
                mask = (batch == i)
                graph_nodes = x[mask].unsqueeze(0)
                cls = self.cls_token.expand(1, -1, -1)
                seq = torch.cat([cls, graph_nodes], dim=1)
                seq = self.transformer(seq)
                graph_feat = seq[:, 0, :]
                graph_features.append(graph_feat)
            
            graph_features = torch.cat(graph_features, dim=0)
            graph_logits = self.graph_classifier(graph_features)
            outputs['graph'] = graph_logits
        
        # 节点分类任务
        if task in ['node', 'both']:
            node_logits = self.node_classifier(x)
            outputs['node'] = node_logits
        
        return outputs


def get_model(model_type='classifier', **kwargs):
    """
    模型工厂函数
    
    参数:
        model_type: 模型类型 ('classifier', 'segmentation', 'hybrid')
        **kwargs: 模型参数
    
    返回:
        model: 模型实例
    """
    if model_type == 'classifier':
        return GCNTransformerClassifier(**kwargs)
    elif model_type == 'segmentation':
        return GCNSegmentation(**kwargs)
    elif model_type == 'hybrid':
        return HybridGCNModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

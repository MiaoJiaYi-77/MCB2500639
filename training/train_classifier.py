"""
训练GCN分类模型 (问题1)
GCN + Transformer架构用于残损分类
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.models import GCNTransformerClassifier
from preprocessing.create_dataset import ContainerGraphDataset


class Trainer:
    """GCN分类模型训练器"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir='checkpoints'
    ):
        """
        初始化训练器
        
        参数:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备 (cuda/cpu)
            learning_rate: 学习率
            weight_decay: 权重衰减
            save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 训练历史 - 添加更多评估指标
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_auc': [],
            'val_balanced_acc': [],
            'val_mcc': [],
            'val_kappa': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='训练')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(all_labels)
        avg_acc = (all_preds == all_labels).mean()
        
        # 计算F1：单类别时也计算该类别的F1
        unique_labels = np.unique(all_labels)
        if len(unique_labels) >= 2:
            avg_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        else:
            # 仅有一个类别时，按该类别作为正类计算F1
            pos = int(unique_labels[0])
            try:
                avg_f1 = f1_score(all_labels, all_preds, average='binary', pos_label=pos, zero_division=0)
            except Exception:
                avg_f1 = 0.0
        
        return avg_loss, avg_acc, avg_f1
    
    @torch.no_grad()
    def validate(self):
        """验证模型 - 添加完整评估指标"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.val_loader, desc='验证')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # 前向传播
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            # 统计
            total_loss += loss.item() * batch.num_graphs
            
            # 预测
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算基础指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        avg_loss = total_loss / len(all_labels)
        avg_acc = (all_preds == all_labels).mean()
        
        # 检查类别数量
        unique_labels = np.unique(all_labels)
        
        # 初始化指标
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'balanced_acc': 0.0,
            'mcc': 0.0,
            'kappa': 0.0
        }
        
        # 如果有两个类别，计算完整指标
        if len(unique_labels) >= 2:
            try:
                metrics['precision'] = precision_score(all_labels, all_preds, average='binary', zero_division=0)
                metrics['recall'] = recall_score(all_labels, all_preds, average='binary', zero_division=0)
                metrics['f1'] = f1_score(all_labels, all_preds, average='binary', zero_division=0)
                metrics['balanced_acc'] = balanced_accuracy_score(all_labels, all_preds)
                metrics['mcc'] = matthews_corrcoef(all_labels, all_preds)
                metrics['kappa'] = cohen_kappa_score(all_labels, all_preds)
                metrics['auc'] = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                print(f"\n⚠️ 警告: 部分指标计算失败: {e}")
        else:
            # 仅一个类别时，按该类别作为正类计算P/R/F1，并给出说明
            pos = int(unique_labels[0])
            try:
                metrics['precision'] = precision_score(all_labels, all_preds, pos_label=pos, average='binary', zero_division=0)
                metrics['recall'] = recall_score(all_labels, all_preds, pos_label=pos, average='binary', zero_division=0)
                metrics['f1'] = f1_score(all_labels, all_preds, pos_label=pos, average='binary', zero_division=0)
            except Exception as e:
                print(f"\n⚠️ 单类别评估计算失败: {e}")
            # 单类别没有负类，Balanced Accuracy 退化为该类别的召回率
            metrics['balanced_acc'] = metrics['recall']
            # AUC/MCC/Kappa 在单类别下不可计算，保持为0并提示
            print("\nℹ️ 提示: 验证集中仅包含一个类别，AUC/MCC/Kappa 无法计算，已置为0.0；Balanced Accuracy=该类别召回率。")
        
        # 打印详细评估报告
        print(f"\n" + "="*70)
        print(f"📊 验证集评估报告:")
        print(f"="*70)
        print(f"\n【基础指标】")
        print(f"  总样本数: {len(all_labels)}")
        print(f"  准确率 (Accuracy): {avg_acc:.4f}")
        print(f"  平衡准确率 (Balanced Accuracy): {metrics['balanced_acc']:.4f}")
        
        print(f"\n【分类性能】")
        print(f"  精确率 (Precision): {metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-Score): {metrics['f1']:.4f}")
        
        print(f"\n【高级指标】")
        print(f"  AUC-ROC: {metrics['auc']:.4f}")
        print(f"  Matthews相关系数 (MCC): {metrics['mcc']:.4f}")
        print(f"  Cohen's Kappa: {metrics['kappa']:.4f}")
        
        # 混淆矩阵
        if len(unique_labels) >= 2:
            cm = confusion_matrix(all_labels, all_preds)
            print(f"\n【混淆矩阵】")
            print(f"  真实\\预测    类别0    类别1")
            for i, row in enumerate(cm):
                print(f"  类别{i}      {row[0]:6d}  {row[1]:6d}")
        
        # 每个类别的详细统计 (保留硬编码循环以适应单类别情况)
        print(f"\n【类别性能】")
        for cls in [0, 1]:
            mask = (all_labels == cls)
            if mask.sum() > 0:  # 只有当该类别存在时才打印
                cls_acc = (all_preds[mask] == cls).sum() / mask.sum()
                cls_name = "无残损" if cls == 0 else "有残损"
                cls_count = mask.sum()
                cls_correct = (all_preds[mask] == cls).sum()
                print(f"  {cls_name} (类别{cls}): {cls_count} 样本, "
                      f"正确 {cls_correct}, 准确率 {cls_acc:.4f}")
        
        print(f"="*70 + "\n")
        
        return avg_loss, avg_acc, metrics
    
    def train(self, num_epochs):
        """
        训练模型
        
        参数:
            num_epochs: 训练轮数
        """
        print("=" * 60)
        print("开始训练GCN分类模型")
        print("=" * 60)
        print(f"设备: {self.device}")
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")
        print(f"总轮数: {num_epochs}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate()
            
            # 更新学习率 (基于F1分数)
            self.scheduler.step(val_metrics['f1'])
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_balanced_acc'].append(val_metrics['balanced_acc'])
            self.history['val_mcc'].append(val_metrics['mcc'])
            self.history['val_kappa'].append(val_metrics['kappa'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印统计
            print(f"\n📈 Epoch {epoch} 结果摘要:")
            print(f"  【训练集】")
            print(f"    Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  【验证集】")
            print(f"    Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"    AUC: {val_metrics['auc']:.4f}, MCC: {val_metrics['mcc']:.4f}, Kappa: {val_metrics['kappa']:.4f}")
            print(f"  【学习状态】")
            print(f"    学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型 (基于F1分数)
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ 保存最佳模型 (F1: {val_metrics['f1']:.4f}, Acc: {val_acc:.4f})")
            
            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "=" * 60)
        print("✅ 训练完成!")
        print("=" * 60)
        print(f"📊 最佳性能指标:")
        print(f"  最佳验证F1: {self.best_val_f1:.4f}")
        print(f"  最佳验证准确率: {self.best_val_acc:.4f}")
        print("=" * 60)
        
        # 保存训练历史
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练历史已保存: {history_path}")


def main():
    """主函数"""
    # 配置 - 使用相对路径
    config = {
        'data_root': Path(__file__).parent.parent / 'preprocessed',
        'batch_size': 24,  # 增大批次，充分利用6GB显存
        'num_epochs': 100,  # 增加训练轮数以充分训练
        'learning_rate': 5e-4,  # 适中的学习率
        'weight_decay': 5e-4,  # 正则化权重
        'hidden_dim': 256,  # 保持平衡的隐藏维度
        'gcn_layers': 3,  # GCN层数
        'num_heads': 8,  # 注意力头数
        'num_transformer_layers': 2,  # Transformer层数
        'dropout': 0.5,  # 增加dropout防止过拟合
        'num_workers': 6,  # 使用6个worker充分利用CPU (16核的1/3左右)
        'save_dir': 'checkpoints_classifier'
    }
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  显存容量: {total_memory:.2f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  PyTorch版本: {torch.__version__}")
        
        # CUDA性能优化设置
        torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
        torch.backends.cudnn.deterministic = False  # 允许非确定性以提速
        torch.cuda.empty_cache()  # 清理显存缓存
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    print(f"  CPU核心数: {cpu_count}")
    print(f"  使用Workers: {config['num_workers']}")
    
    print(f"\n⚙️  训练配置:")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  最大轮数: {config['num_epochs']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  Dropout: {config['dropout']}")
    print("=" * 80)
    
    # 创建数据集
    print("\n加载数据集...")
    
    # 使用已划分的训练集和验证集
    train_dataset = ContainerGraphDataset(root=config['data_root'], split='train')
    val_dataset = ContainerGraphDataset(root=config['data_root'], split='val')
    
    print(f"训练集: {len(train_dataset)} 个图")
    print(f"验证集: {len(val_dataset)} 个图")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config['num_workers'] > 0 else False,  # 保持workers活跃
        prefetch_factor=2 if config['num_workers'] > 0 else None  # 预取数据
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # 验证时可以用更大的batch
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        prefetch_factor=2 if config['num_workers'] > 0 else None
    )
    
    # 创建模型
    print("\n创建模型...")
    model = GCNTransformerClassifier(
        node_features=2048,
        hidden_dim=config['hidden_dim'],
        gcn_layers=config['gcn_layers'],
        num_heads=config['num_heads'],
        num_transformer_layers=config['num_transformer_layers'],
        num_classes=2,
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 保存配置
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # 将Path对象转换为字符串以便JSON序列化
    config_to_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存: {config_path}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir=config['save_dir']
    )
    
    # 开始训练
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()

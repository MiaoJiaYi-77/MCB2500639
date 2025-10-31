"""
独立推断脚本：在 preprocessed/val 上运行模型推断并保存结果为 CSV
用法示例：
    python training/infer_val.py --checkpoint checkpoints_classifier/best_model.pth --output_csv test_result.csv
"""
import argparse
from pathlib import Path
import sys
import json
import csv
from tqdm import tqdm
import torch
import numpy as np

# ensure repo root on path
sys.path.append(str(Path(__file__).parent.parent))

from models.models import GCNTransformerClassifier
from preprocessing.create_dataset import ContainerGraphDataset
from torch_geometric.loader import DataLoader


def load_model(checkpoint_path, device):
    # 这里构建模型时需与训练时保持一致
    model = GCNTransformerClassifier(
        node_features=2048,
        hidden_dim=256,
        gcn_layers=3,
        num_heads=8,
        num_transformer_layers=2,
        num_classes=2,
        dropout=0.5
    )
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)

    # 数据集
    dataset = ContainerGraphDataset(root=data_root, split='val')
    print(f"加载数据: {len(dataset)} 个样本 (split=val)")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # 模型
    model = load_model(args.checkpoint, device)

    image_names = []
    preds = []
    probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='推断 - val'):
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            imgs = getattr(batch, 'image_name', None)
            if imgs is not None:
                if isinstance(imgs, (list, tuple)):
                    image_names.extend(imgs)
                else:
                    try:
                        image_names.extend(list(imgs))
                    except Exception:
                        image_names.append(str(imgs))
            else:
                # 退化到索引编号
                image_names.extend([f"idx_{i}" for i in range(pred.size(0))])

            preds.extend(pred.cpu().numpy().tolist())
            # 记录每张图的最大概率
            if prob.size(1) > 1:
                probs.extend(prob.max(dim=1).values.cpu().numpy().tolist())
            else:
                probs.extend(prob[:, 0].cpu().numpy().tolist())

    # 保存为 CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred', 'prob'])
        for img, p, pr in zip(image_names, preds, probs):
            writer.writerow([img, int(p), float(pr)])

    print(f"推断完成，输出保存到: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_classifier/best_model.pth', help='模型 checkpoint 路径')
    parser.add_argument('--data_root', type=str, default=str(Path(__file__).parent.parent / 'preprocessed'), help='preprocessed 数据根目录')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--output_csv', type=str, default='test_result.csv')
    args = parser.parse_args()

    infer(args)

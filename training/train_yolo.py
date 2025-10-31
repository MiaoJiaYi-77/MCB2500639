"""
YOLO11 训练启动脚本 - 适配 RTX 4050 6GB 显存
使用方法：python train_yolo.py
"""

import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import yaml
from pathlib import Path
import warnings
import csv
from datetime import datetime
import numpy as np

if __name__ == '__main__':
    # 读取配置文件
    with open('args.yaml', 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)

    # 读取 data.yaml 并调整 split：
    # 我们的约定是：train 有标签、test 有标签（用于训练时的调参）、val 无标签（用于最终提交）
    data_path = Path(args['data'])
    try:
        with open(data_path, 'r', encoding='utf-8') as df:
            data_cfg = yaml.safe_load(df)
    except Exception as e:
        raise RuntimeError(f"无法读取 data 文件 {data_path}: {e}")

    # 生成一个临时 data yaml，用于训练时将 val 指向原始 test（有标签）
    tmp_data = dict(data_cfg)
    # 保留 train，交换 val/test：把 val 指向原始的 test（用于训练期间验证/调参）
    if 'test' in data_cfg and 'val' in data_cfg:
        tmp_data['train'] = data_cfg.get('train')
        tmp_data['val'] = data_cfg.get('test')
        tmp_data['test'] = data_cfg.get('val')
    else:
        warnings.warn("data.yaml 中未同时包含 'test' 和 'val' 字段，使用原始 data 配置进行训练。")
        tmp_data = data_cfg

    tmp_path = Path(args.get('project', '.')) / f"data_for_training_{Path(args['data']).stem}.yaml"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, 'w', encoding='utf-8') as tf:
        yaml.safe_dump(tmp_data, tf, sort_keys=False, allow_unicode=True)

    # 创建新模型
    model = YOLO(args['model'])

    # 开始训练（使用配置文件中的参数）
    results = model.train(
        # 使用临时 data 配置，使训练时的验证（val）为原始 test（有标签）
        data=str(tmp_path),
        epochs=args['epochs'],
        batch=args['batch'],
        imgsz=args['imgsz'],
        device=args['device'],
        workers=args['workers'],
        amp=args['amp'],
        project=args['project'],
        name=args['name'],
        patience=args['patience'],
        save=args['save'],
        plots=args['plots'],
        resume=args['resume'],  # ✅ 关键：传递 resume 参数
        
        # 优化器设置
        optimizer=args['optimizer'],
        lr0=args['lr0'],
        lrf=args['lrf'],
        momentum=args['momentum'],
        weight_decay=args['weight_decay'],
        
        # 数据增强
        mosaic=args['mosaic'],
        mixup=args['mixup'],
        hsv_h=args['hsv_h'],
        hsv_s=args['hsv_s'],
        hsv_v=args['hsv_v'],
        degrees=args['degrees'],
        translate=args['translate'],
        scale=args['scale'],
        fliplr=args['fliplr'],
        
        # 其他设置
        verbose=args['verbose'],
        seed=args['seed'],
        deterministic=args['deterministic'],
    )

    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print(f"最后模型保存在: {results.save_dir}/weights/last.pt")
    print("="*60)

    # 训练完成后：对原始 val（无标签）进行推断并保存为 CSV（用于提交/上报）
    try:
        orig_data_path = data_path
        orig_data = data_cfg
        val_rel = orig_data.get('val')
        if val_rel is None:
            warnings.warn('原始 data.yaml 未定义 val 路径，跳过最终推断。')
        else:
            val_images_dir = (orig_data_path.parent / val_rel).resolve()
            print(f"\n开始对无标签验证集 ({val_images_dir}) 进行推断并保存预测...")

            preds = model.predict(
                source=str(val_images_dir),
                imgsz=args.get('imgsz', 640),
                device=args.get('device', '0'),
                conf=args.get('conf', 0.25),
                iou=args.get('iou', 0.45),
                max_det=args.get('max_det', 300)
            )

            # 保存预测到 CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = Path(args.get('project', '.'))
            save_dir.mkdir(parents=True, exist_ok=True)
            preds_file = save_dir / f'test_result.csv'
            # 覆盖已有文件
            with open(preds_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_path', 'class', 'conf', 'x1', 'y1', 'x2', 'y2'])

                for r in preds:
                    # 获取源图像路径（不同 ultralytics 版本属性名可能不同）
                    img_path = getattr(r, 'path', None)
                    if img_path is None:
                        # 尝试从 r.orig_img 或 r.orig_img_path 获取信息
                        img_path = getattr(r, 'orig_img', None)

                    # r.boxes 可能为空或格式不同；统一读取列表形式
                    try:
                        boxes = r.boxes
                        if boxes is None or len(boxes) == 0:
                            continue

                        # xyxy, conf, cls arrays
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy()

                        # 按置信度排序并取前4个（严重程度 = 置信度）
                        order = np.argsort(-confs)
                        topk = order[:4]

                        for idx in topk:
                            b = xyxy[idx]
                            c = confs[idx]
                            cl = clss[idx]
                            x1, y1, x2, y2 = b.tolist()
                            writer.writerow([str(img_path), int(cl), float(c), float(x1), float(y1), float(x2), float(y2)])
                    except Exception:
                        # 无检测结果或 API 不一致时跳过该图
                        continue

            print(f"预测已保存: {preds_file}")
    except Exception as e:
        warnings.warn(f"保存最终验证集预测时出现错误: {e}")

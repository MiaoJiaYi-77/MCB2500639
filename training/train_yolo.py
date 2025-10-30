"""
YOLO11 训练启动脚本 - 适配 RTX 4050 6GB 显存
使用方法：python train_yolo.py
"""

import os
# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import yaml

if __name__ == '__main__':
    # 读取配置文件
    with open('args.yaml', 'r', encoding='utf-8') as f:
        args = yaml.safe_load(f)

    
    # 创建新模型
    model = YOLO(args['model'])

    # 开始训练（使用配置文件中的参数）
    results = model.train(
        data=args['data'],
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

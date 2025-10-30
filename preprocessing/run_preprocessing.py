"""
预处理主流程脚本
按顺序执行所有预处理步骤

使用方法:
    python run_preprocessing.py
    python run_preprocessing.py --skip-superpixel  # 跳过超像素分割
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
from datetime import datetime

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class PreprocessingPipeline:
    """预处理流水线"""
    
    def __init__(self, data_root='data', output_root='preprocessed'):
        """
        初始化预处理流水线
        
        参数:
            data_root: 原始数据根目录
            output_root: 预处理输出根目录
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.preprocessing_dir = Path(__file__).parent
        
        print("="*80)
        print("预处理流水线初始化")
        print("="*80)
        print(f"数据根目录: {self.data_root.absolute()}")
        print(f"输出根目录: {self.output_root.absolute()}")
        print(f"预处理脚本目录: {self.preprocessing_dir.absolute()}")
        print("="*80 + "\n")
    
    def run_step(self, script_name, description, extra_args=None):
        """
        运行预处理步骤
        
        参数:
            script_name: 脚本文件名
            description: 步骤描述
        """
        print("\n" + "="*80)
        print(f"步骤: {description}")
        print("="*80)
        print(f"执行脚本: {script_name}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80)
        
        script_path = self.preprocessing_dir / script_name
        
        if not script_path.exists():
            print(f"❌ 错误: 脚本不存在: {script_path}")
            return False
        
        # 执行脚本
        try:
            # 组装命令
            cmd = [sys.executable, str(script_path)]
            if extra_args:
                cmd.extend(extra_args)

            result = subprocess.run(
                cmd,
                cwd=self.preprocessing_dir.parent,  # 在项目根目录执行
                check=True,
                capture_output=False  # 显示输出
            )
            
            print("-"*80)
            print(f"✓ 步骤完成: {description}")
            print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
            
        except subprocess.CalledProcessError as e:
            print("-"*80)
            print(f"❌ 错误: {description} 失败")
            print(f"返回码: {e.returncode}")
            return False
        except Exception as e:
            print("-"*80)
            print(f"❌ 错误: {str(e)}")
            return False
    
    def run_all(self, skip_steps=None, splits=None):
        """
        运行完整的预处理流程
        
        参数:
            skip_steps: 要跳过的步骤列表
        """
        if skip_steps is None:
            skip_steps = []
        
        print("\n" + "="*80)
        print("开始完整预处理流程")
        print("="*80)
        
        start_time = datetime.now()
        
        # 定义所有预处理步骤
        steps = [
            ('preprocess_superpixel.py', '超像素分割', 'superpixel'),
            ('build_graph.py', '构建图结构', 'graph'),
            ('extract_features.py', '提取ResNet特征', 'features'),
            ('create_dataset.py', '创建PyG数据集', 'dataset'),
        ]
        
        success_count = 0
        failed_steps = []
        
        for script, description, step_id in steps:
            if step_id in skip_steps:
                print(f"\n⏭️  跳过步骤: {description}")
                continue
            
            # 为各步骤构建参数
            extra_args = []
            # 统一将 split 列表传入子脚本（子脚本将同时兼容 --split 与 --splits）
            if splits:
                if len(splits) == 1:
                    extra_args += ["--split", splits[0]]
                else:
                    extra_args += ["--splits", ",".join(splits)]

            if step_id == 'superpixel':
                # 传入数据根与输出根
                extra_args += [
                    "--data-root", str(self.data_root),
                    "--output-root", str(self.output_root),
                ]
            elif step_id == 'graph':
                # 传入预处理根目录
                extra_args += [
                    "--preprocessed-root", str(self.output_root),
                ]
            elif step_id == 'features':
                # 传入图像数据根与预处理根/输出根
                extra_args += [
                    "--data-root", str(self.data_root),
                    "--preprocessed-root", str(self.output_root),
                    "--output-root", str(self.output_root),
                ]
                # 为避免在某些环境下自动估算batch size卡住，默认禁用自动估算并使用小的batch_size
                # 这样可以保证特征提取步骤能够在低显存或有问题的环境下继续执行。
                extra_args += ["--no-auto-batch", "--batch-size", "1"]
            elif step_id == 'dataset':
                # 传入预处理根目录
                extra_args += [
                    "--preprocessed-root", str(self.output_root),
                ]

            success = self.run_step(script, description, extra_args=extra_args)
            
            if success:
                success_count += 1
            else:
                failed_steps.append(description)
                print(f"\n⚠️  步骤失败,但继续执行...")
        
        # 总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("预处理流程完成!")
        print("="*80)
        print(f"总步骤数: {len(steps) - len(skip_steps)}")
        print(f"成功: {success_count}")
        print(f"失败: {len(failed_steps)}")
        if failed_steps:
            print(f"失败步骤: {', '.join(failed_steps)}")
        print(f"总耗时: {duration}")
        print("="*80)
        
        # 检查输出目录结构
        self.verify_output()
    
    def verify_output(self):
        """验证预处理输出"""
        print("\n验证预处理输出:")
        print("-"*80)
        
        required_dirs = [
            self.output_root / 'superpixels',
            self.output_root / 'graphs',
            self.output_root / 'node_features',
            self.output_root / 'processed',
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                # 统计子目录数量
                subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                print(f"✓ {dir_path.name}: {len(subdirs)} 个子目录")
            else:
                print(f"❌ {dir_path.name}: 不存在")
        
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description='运行完整的预处理流程')
    parser.add_argument('--data-root', type=str, default='data',
                        help='原始数据根目录')
    parser.add_argument('--output-root', type=str, default='preprocessed',
                        help='预处理输出根目录')
    parser.add_argument('--split', type=str, default=None,
                        help='单个数据集分割名称（如 test_noisy）')
    parser.add_argument('--splits', type=str, default=None,
                        help='以逗号分隔的多个分割名称，形如 "train,val,test"')
    parser.add_argument('--skip-superpixel', action='store_true',
                        help='跳过超像素分割步骤')
    parser.add_argument('--skip-graph', action='store_true',
                        help='跳过图构建步骤')
    parser.add_argument('--skip-features', action='store_true',
                        help='跳过特征提取步骤')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='跳过数据集创建步骤')
    
    args = parser.parse_args()
    
    # 解析 splits
    splits = None
    if args.split:
        splits = [args.split]
    elif args.splits:
        splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    # 确定要跳过的步骤
    skip_steps = []
    if args.skip_superpixel:
        skip_steps.append('superpixel')
    if args.skip_graph:
        skip_steps.append('graph')
    if args.skip_features:
        skip_steps.append('features')
    if args.skip_dataset:
        skip_steps.append('dataset')
    
    # 创建并运行流水线
    pipeline = PreprocessingPipeline(
        data_root=args.data_root,
        output_root=args.output_root
    )
    
    pipeline.run_all(skip_steps=skip_steps, splits=splits)


if __name__ == '__main__':
    main()

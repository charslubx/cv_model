#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片推理演示脚本

展示如何使用训练好的模型对图片进行分类推理
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference import FashionInference

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='服装图片分类推理演示')
    parser.add_argument('--model', '-m', type=str, default='mixed_checkpoints/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--image', '-i', type=str, required=False,
                       help='输入图片路径')
    parser.add_argument('--batch', '-b', type=str, nargs='+', required=False,
                       help='批量输入图片路径')
    parser.add_argument('--output', '-o', type=str, required=False,
                       help='输出结果文件路径')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='显示详细结果')
    parser.add_argument('--auto-find', '-a', action='store_true',
                       help='自动从数据集中寻找测试图片')
    
    args = parser.parse_args()
    
    try:
        # 检查模型文件
        if not os.path.exists(args.model):
            logger.error(f"模型文件不存在: {args.model}")
            logger.info("请先训练模型或提供正确的模型路径")
            logger.info("使用 --model 参数指定模型路径")
            return 1
        
        # 创建推理器
        logger.info(f"加载模型: {args.model}")
        inferencer = FashionInference(args.model)
        
        # 确定要处理的图片
        image_paths = []
        
        if args.image:
            # 单张图片
            if os.path.exists(args.image):
                image_paths = [args.image]
            else:
                logger.error(f"图片文件不存在: {args.image}")
                return 1
                
        elif args.batch:
            # 批量图片
            for img_path in args.batch:
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                else:
                    logger.warning(f"图片文件不存在，跳过: {img_path}")
                    
        elif args.auto_find:
            # 自动寻找测试图片
            logger.info("自动寻找测试图片...")
            
            search_dirs = [
                "/home/cv_model/fabric/train",
                "/home/cv_model/fiber/train"
            ]
            
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for class_dir in os.listdir(search_dir)[:3]:  # 每个数据集取3个类别
                        class_path = os.path.join(search_dir, class_dir)
                        if os.path.isdir(class_path):
                            images = [f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            if images:
                                image_paths.append(os.path.join(class_path, images[0]))
                                if len(image_paths) >= 5:  # 最多找5张图片
                                    break
                    if len(image_paths) >= 5:
                        break
            
            if not image_paths:
                logger.warning("未找到测试图片")
                logger.info("请使用 --image 或 --batch 参数指定图片路径")
                return 1
                
        else:
            # 没有指定图片
            logger.error("请指定要处理的图片")
            logger.info("使用方法:")
            logger.info("  单张图片: python demo_inference.py --image path/to/image.jpg")
            logger.info("  批量图片: python demo_inference.py --batch path1.jpg path2.jpg")
            logger.info("  自动寻找: python demo_inference.py --auto-find")
            return 1
        
        logger.info(f"将处理 {len(image_paths)} 张图片")
        
        # 处理图片
        all_results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"\n处理图片 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # 进行推理
                results = inferencer.predict(image_path)
                results['image_path'] = image_path
                all_results.append(results)
                
                # 显示结果
                print(f"\n{'='*80}")
                print(f"图片: {image_path}")
                formatted_results = inferencer.format_results(results, detailed=args.detailed)
                print(formatted_results)
                
            except Exception as e:
                logger.error(f"处理图片失败 {image_path}: {e}")
                all_results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # 保存结果到文件（如果指定）
        if args.output:
            try:
                import json
                
                # 准备保存的数据
                save_data = []
                for result in all_results:
                    if 'error' not in result:
                        # 转换tensor为列表以便JSON序列化
                        save_result = {
                            'image_path': result['image_path'],
                            'predictions': result['predictions']
                        }
                        save_data.append(save_result)
                    else:
                        save_data.append({
                            'image_path': result['image_path'],
                            'error': result['error']
                        })
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"结果已保存到: {args.output}")
                
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
        
        # 统计总结
        successful = sum(1 for r in all_results if 'error' not in r)
        failed = len(all_results) - successful
        
        print(f"\n{'='*80}")
        print("处理总结:")
        print(f"  总计: {len(all_results)} 张图片")
        print(f"  成功: {successful} 张")
        print(f"  失败: {failed} 张")
        
        if successful > 0:
            print("\n推理功能正常工作！")
            return 0
        else:
            print("\n所有图片处理失败，请检查模型和图片")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

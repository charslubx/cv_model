"""
快速推理脚本 - 用于快速对图片进行推理
使用方法：
    # 基础用法（自动从数据集加载属性）
    python quick_inference.py --image path/to/image.jpg --model path/to/model.pth
    
    # 指定属性文件
    python quick_inference.py --image path/to/image.jpg --model path/to/model.pth \
        --attr-file /path/to/list_attr_cloth.txt
    
    # 批量推理
    python quick_inference.py --images path/to/images/ --model path/to/model.pth
"""

import torch
import argparse
import os
import sys
from inference import FashionInferenceWrapper, load_attr_names_from_file
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='服装属性推理工具')
    
    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径 (*.pth)')
    
    # 输入选项（二选一）
    parser.add_argument('--image', type=str,
                        help='单张图片路径')
    parser.add_argument('--images', type=str,
                        help='图片目录路径（批量推理）')
    
    # 可选参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='属性判定阈值 (默认: 0.5)')
    parser.add_argument('--output', type=str, default='inference_result.json',
                        help='输出JSON文件路径 (默认: inference_result.json)')
    parser.add_argument('--attr-file', type=str,
                        help='属性定义文件路径 (list_attr_cloth.txt)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='运行设备 (默认: auto)')
    parser.add_argument('--raw', action='store_true',
                        help='输出原始logits和概率值')
    parser.add_argument('--no-summary', action='store_true',
                        help='不打印摘要信息')
    
    # 纹理分类相关
    parser.add_argument('--textile', action='store_true',
                        help='启用纹理分类')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 1. 验证输入
    if not args.image and not args.images:
        logger.error("错误: 必须指定 --image 或 --images 参数")
        sys.exit(1)
    
    if args.image and args.images:
        logger.error("错误: --image 和 --images 不能同时使用")
        sys.exit(1)
    
    # 2. 检查模型文件
    if not os.path.exists(args.model):
        logger.error(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 3. 加载模型
    logger.info(f"加载模型: {args.model}")
    try:
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        model = torch.load(args.model, map_location=device)
        model.eval()
        logger.info(f"模型加载成功，运行设备: {device}")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        sys.exit(1)
    
    # 4. 加载属性名称
    attr_names = None
    
    if args.attr_file:
        # 用户指定了属性文件
        if os.path.exists(args.attr_file):
            attr_names = load_attr_names_from_file(args.attr_file)
            logger.info(f"从指定文件加载了 {len(attr_names)} 个属性名称")
        else:
            logger.error(f"指定的属性文件不存在: {args.attr_file}")
            sys.exit(1)
    else:
        # 尝试自动加载
        from inference import get_attr_names_from_dataset
        
        logger.info("未指定属性文件，尝试从数据集自动加载...")
        attr_names = get_attr_names_from_dataset()
        
        if attr_names:
            logger.info(f"成功自动加载了 {len(attr_names)} 个属性名称")
        else:
            logger.warning("自动加载失败，使用模型内部默认值")
            logger.warning("建议使用 --attr-file 参数指定属性定义文件以获得准确的属性名称")
    
    # 5. 创建推理包装器
    wrapper = FashionInferenceWrapper(
        model=model,
        attr_names=attr_names,
        threshold=args.threshold,
        device=device,
        enable_textile_classification=args.textile
    )
    
    # 6. 执行推理
    if args.image:
        # 单张图片推理
        if not os.path.exists(args.image):
            logger.error(f"错误: 图片文件不存在: {args.image}")
            sys.exit(1)
        
        logger.info(f"\n开始推理图片: {args.image}")
        logger.info("-" * 60)
        
        result = wrapper.predict_and_save(
            args.image,
            args.output,
            return_raw=args.raw
        )
        
        # 打印摘要
        if not args.no_summary:
            print("\n" + "=" * 60)
            print("推理结果")
            print("=" * 60)
            print(wrapper.get_summary(result))
            print("=" * 60)
        
        logger.info(f"\n完整结果已保存到: {args.output}")
    
    else:
        # 批量推理
        if not os.path.exists(args.images):
            logger.error(f"错误: 图片目录不存在: {args.images}")
            sys.exit(1)
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.images, ext)))
        
        if not image_paths:
            logger.error(f"错误: 在 {args.images} 中没有找到图片文件")
            sys.exit(1)
        
        logger.info(f"\n找到 {len(image_paths)} 张图片")
        logger.info(f"开始批量推理...")
        logger.info("-" * 60)
        
        results = wrapper.predict_batch_and_save(
            image_paths,
            args.output,
            return_raw=args.raw
        )
        
        # 打印摘要
        if not args.no_summary:
            print("\n" + "=" * 60)
            print(f"批量推理结果 (共 {len(results)} 张图片)")
            print("=" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n[{i}/{len(results)}]")
                print(wrapper.get_summary(result))
                print("-" * 50)
        
        logger.info(f"\n完整结果已保存到: {args.output}")
    
    logger.info("\n推理完成！")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n用户中断执行")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

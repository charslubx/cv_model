"""
推理示例脚本 - 演示如何使用推理包装器进行图片分类
"""

import torch
import os
import sys
from inference import FashionInferenceWrapper, load_attr_names_from_file
from base_model import FullModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_single_image_inference():
    """示例1：单张图片推理"""
    logger.info("=" * 60)
    logger.info("示例1：单张图片推理")
    logger.info("=" * 60)
    
    # 1. 加载模型
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先训练模型或提供正确的模型路径")
        return
    
    logger.info(f"加载模型: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 2. 准备属性名称（从文件加载或使用默认值）
    attr_file = "/home/cv_model/deepfashion/Category and Attribute Prediction Benchmark/Anno_fine/list_attr_cloth.txt"
    
    if os.path.exists(attr_file):
        attr_names = load_attr_names_from_file(attr_file)
        logger.info(f"从文件加载了 {len(attr_names)} 个属性名称")
    else:
        logger.warning("属性定义文件不存在，使用默认属性名称")
        attr_names = None  # 使用默认值
    
    # 3. 创建推理包装器
    wrapper = FashionInferenceWrapper(
        model=model,
        attr_names=attr_names,
        threshold=0.5,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # 4. 推理单张图片
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"测试图片不存在: {image_path}")
        logger.info("请提供有效的图片路径")
        return
    
    logger.info(f"正在推理图片: {image_path}")
    result = wrapper.predict_single(image_path, return_raw=False)
    
    # 5. 打印结果
    print("\n" + wrapper.get_summary(result))
    
    # 6. 保存结果到JSON
    wrapper.predict_and_save(image_path, "result_single.json", return_raw=True)
    logger.info("详细结果已保存到: result_single.json")


def example_batch_inference():
    """示例2：批量图片推理"""
    logger.info("\n" + "=" * 60)
    logger.info("示例2：批量图片推理")
    logger.info("=" * 60)
    
    # 1. 加载模型
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 2. 创建推理包装器
    wrapper = FashionInferenceWrapper(
        model=model,
        threshold=0.5
    )
    
    # 3. 批量推理
    image_dir = "test_images"
    if not os.path.exists(image_dir):
        logger.error(f"图片目录不存在: {image_dir}")
        return
    
    # 获取所有图片
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_paths:
        logger.error("没有找到图片文件")
        return
    
    logger.info(f"找到 {len(image_paths)} 张图片")
    
    # 批量推理
    results = wrapper.predict_batch(image_paths, return_raw=False)
    
    # 4. 打印每张图片的结果
    for result in results:
        print("\n" + "=" * 50)
        print(wrapper.get_summary(result))
    
    # 5. 保存批量结果
    wrapper.predict_batch_and_save(image_paths, "results_batch.json", return_raw=True)
    logger.info("批量结果已保存到: results_batch.json")


def example_with_textile_classification():
    """示例3：包含纹理分类的推理"""
    logger.info("\n" + "=" * 60)
    logger.info("示例3：包含纹理分类的推理")
    logger.info("=" * 60)
    
    # 1. 加载启用了纹理分类的模型
    model_path = "checkpoints/best_model_textile.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先训练包含纹理分类的模型")
        return
    
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # 2. 准备Fabric和Fiber类别名称（示例）
    fabric_names = [
        'canvas', 'denim', 'flannel', 'gauze', 'lace', 'leather',
        'linen', 'satin', 'silk', 'velvet', 'wool', 'chiffon',
        'cotton', 'knit', 'tweed', 'jersey', 'fleece', 'corduroy',
        'mesh', 'nylon'
    ]
    
    fiber_names = [
        'cotton', 'polyester', 'nylon', 'silk', 'wool', 'linen',
        'acrylic', 'rayon', 'spandex', 'cashmere', 'mohair', 'angora',
        'bamboo', 'modal', 'lyocell', 'acetate', 'viscose', 'hemp',
        'jute', 'ramie', 'alpaca', 'camel', 'kevlar', 'elastane',
        'microfiber', 'polyamide', 'polypropylene', 'aramid', 'nomex',
        'lurex', 'metallic', 'latex'
    ]
    
    # 3. 创建推理包装器（启用纹理分类）
    wrapper = FashionInferenceWrapper(
        model=model,
        fabric_names=fabric_names,
        fiber_names=fiber_names,
        threshold=0.5,
        enable_textile_classification=True
    )
    
    # 4. 推理
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"测试图片不存在: {image_path}")
        return
    
    result = wrapper.predict_single(image_path, return_raw=True)
    
    # 5. 打印结果
    print("\n" + wrapper.get_summary(result))
    
    # 6. 保存结果
    wrapper.predict_and_save(image_path, "result_with_textile.json", return_raw=True)


def example_custom_threshold():
    """示例4：使用自定义阈值"""
    logger.info("\n" + "=" * 60)
    logger.info("示例4：使用自定义阈值")
    logger.info("=" * 60)
    
    # 加载模型
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    model = torch.load(model_path, map_location='cpu')
    
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"测试图片不存在: {image_path}")
        return
    
    # 尝试不同的阈值
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        logger.info(f"\n使用阈值: {threshold}")
        
        wrapper = FashionInferenceWrapper(
            model=model,
            threshold=threshold
        )
        
        result = wrapper.predict_single(image_path)
        print(f"\n阈值={threshold} 的结果:")
        print(f"检测到 {result['attributes']['count']} 个属性")
        print(f"属性: {', '.join(result['attributes']['predicted'])}")


def example_programmatic_usage():
    """示例5：在代码中使用推理结果"""
    logger.info("\n" + "=" * 60)
    logger.info("示例5：在代码中使用推理结果")
    logger.info("=" * 60)
    
    # 加载模型
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    model = torch.load(model_path, map_location='cpu')
    wrapper = FashionInferenceWrapper(model=model)
    
    image_path = "test_image.jpg"
    if not os.path.exists(image_path):
        logger.error(f"测试图片不存在: {image_path}")
        return
    
    # 推理
    result = wrapper.predict_single(image_path, return_raw=True)
    
    # 在代码中使用结果
    attrs = result['attributes']
    
    # 检查是否有特定属性
    if 'floral' in attrs['predicted']:
        print("✓ 这件衣服有花纹图案")
    
    if 'long_sleeve' in attrs['predicted']:
        print("✓ 这是长袖款式")
    
    # 获取最有信心的3个属性
    top_3 = list(attrs['confidence_scores'].items())[:3]
    print(f"\n最有信心的3个属性:")
    for attr, conf in top_3:
        print(f"  - {attr}: {conf:.2%}")
    
    # 统计各类型属性
    print(f"\n总共检测到 {attrs['count']} 个属性")
    
    # 获取所有属性的置信度（如果return_raw=True）
    if 'all_scores' in attrs:
        all_scores = attrs['all_scores']
        print(f"\n所有属性的平均置信度: {sum(all_scores.values())/len(all_scores):.2%}")


def main():
    """主函数：运行所有示例"""
    print("=" * 60)
    print("服装属性推理示例脚本")
    print("=" * 60)
    
    # 检查是否有可用的模型
    model_paths = [
        "checkpoints/best_model.pth",
        "best_model.pth",
        "model.pth"
    ]
    
    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            model_found = True
            logger.info(f"找到模型文件: {path}")
            break
    
    if not model_found:
        logger.warning("未找到模型文件，某些示例可能无法运行")
        logger.info("请先训练模型或提供正确的模型路径")
    
    # 运行示例（取消注释想要运行的示例）
    
    # example_single_image_inference()
    # example_batch_inference()
    # example_with_textile_classification()
    # example_custom_threshold()
    # example_programmatic_usage()
    
    logger.info("\n" + "=" * 60)
    logger.info("示例脚本运行完成")
    logger.info("取消注释 main() 中的函数调用来运行特定示例")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

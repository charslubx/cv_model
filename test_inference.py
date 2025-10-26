#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理功能测试脚本

测试图片加载、预处理和模型推理功能
"""

import os
import sys
import torch
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference import FashionInference
from base_model import FullModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_model():
    """创建一个虚拟模型用于测试"""
    logger.info("创建虚拟模型用于测试...")
    
    try:
        model = FullModel(
            num_classes=26,
            enable_segmentation=False,
            enable_textile_classification=True,
            num_fabric_classes=20,
            num_fiber_classes=32,
            gat_dims=[512, 256],  # 使用较小的维度
            gat_heads=4
        )
        
        # 保存虚拟模型
        dummy_model_path = "dummy_model.pth"
        torch.save(model, dummy_model_path)
        logger.info(f"✓ 虚拟模型已保存: {dummy_model_path}")
        
        return dummy_model_path
        
    except Exception as e:
        logger.error(f"创建虚拟模型失败: {e}")
        return None


def test_image_loading():
    """测试图片加载功能"""
    logger.info("测试图片加载功能...")
    
    # 创建测试图片
    test_image_path = "test_image.jpg"
    
    try:
        # 创建一个简单的测试图片
        from PIL import Image
        import numpy as np
        
        # 创建随机图片
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        pil_image.save(test_image_path)
        
        logger.info(f"✓ 测试图片已创建: {test_image_path}")
        
        # 测试图片加载
        image = Image.open(test_image_path).convert('RGB')
        logger.info(f"✓ 图片加载成功: {image.size}")
        
        return test_image_path
        
    except Exception as e:
        logger.error(f"图片加载测试失败: {e}")
        return None


def test_inference_class():
    """测试推理类的基本功能"""
    logger.info("测试推理类的基本功能...")
    
    # 创建虚拟模型
    model_path = create_dummy_model()
    if not model_path:
        logger.error("无法创建虚拟模型")
        return False
    
    # 创建测试图片
    image_path = test_image_loading()
    if not image_path:
        logger.error("无法创建测试图片")
        return False
    
    try:
        # 创建推理器
        inferencer = FashionInference(model_path)
        logger.info("✓ 推理器创建成功")
        
        # 测试图片预处理
        image_tensor = inferencer.load_image(image_path)
        logger.info(f"✓ 图片预处理成功: {image_tensor.shape}")
        
        # 测试推理
        results = inferencer.predict(image_path)
        logger.info("✓ 推理成功")
        
        # 检查结果结构
        expected_keys = ['raw_outputs', 'predictions', 'probabilities', 'top_predictions']
        for key in expected_keys:
            if key in results:
                logger.info(f"  ✓ 结果包含 {key}")
            else:
                logger.warning(f"  ✗ 结果缺少 {key}")
        
        # 格式化输出
        formatted = inferencer.format_results(results)
        logger.info("✓ 结果格式化成功")
        print("\n" + "="*50)
        print("测试推理结果:")
        print(formatted)
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"推理类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理测试文件
        for file_path in [model_path, image_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"✓ 已清理测试文件: {file_path}")
                except:
                    pass


def test_real_data():
    """测试真实数据（如果存在）"""
    logger.info("测试真实数据...")
    
    # 寻找真实图片
    test_dirs = [
        "/home/cv_model/fabric/train",
        "/home/cv_model/fiber/train"
    ]
    
    real_images = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for class_dir in os.listdir(test_dir)[:2]:  # 取前2个类别
                class_path = os.path.join(test_dir, class_dir)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if images:
                        real_images.append(os.path.join(class_path, images[0]))
                        break
    
    if not real_images:
        logger.warning("没有找到真实图片，跳过真实数据测试")
        return True
    
    logger.info(f"找到 {len(real_images)} 张真实图片")
    
    # 创建虚拟模型进行测试
    model_path = create_dummy_model()
    if not model_path:
        logger.error("无法创建虚拟模型")
        return False
    
    try:
        # 创建推理器
        inferencer = FashionInference(model_path)
        
        # 测试真实图片
        for i, image_path in enumerate(real_images[:3]):  # 只测试前3张
            logger.info(f"测试真实图片 {i+1}: {image_path}")
            
            try:
                results = inferencer.predict(image_path)
                logger.info(f"✓ 真实图片 {i+1} 推理成功")
                
                # 显示简化结果
                if 'predictions' in results:
                    for task, pred in results['predictions'].items():
                        if pred:
                            logger.info(f"  {task}: {pred}")
                
            except Exception as e:
                logger.error(f"真实图片 {i+1} 推理失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"真实数据测试失败: {e}")
        return False
    
    finally:
        # 清理测试文件
        if model_path and os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass


def test_batch_inference():
    """测试批量推理"""
    logger.info("测试批量推理...")
    
    # 创建多个测试图片
    test_images = []
    for i in range(3):
        image_path = f"test_batch_{i}.jpg"
        try:
            # 创建随机图片
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_image = Image.fromarray(test_image)
            pil_image.save(image_path)
            test_images.append(image_path)
        except Exception as e:
            logger.error(f"创建测试图片 {i} 失败: {e}")
    
    if not test_images:
        logger.error("无法创建测试图片")
        return False
    
    # 创建虚拟模型
    model_path = create_dummy_model()
    if not model_path:
        logger.error("无法创建虚拟模型")
        return False
    
    try:
        # 创建推理器
        inferencer = FashionInference(model_path)
        
        # 批量推理
        batch_results = inferencer.predict_batch(test_images)
        logger.info(f"✓ 批量推理成功: {len(batch_results)} 个结果")
        
        # 检查结果
        for i, result in enumerate(batch_results):
            if 'error' in result:
                logger.error(f"批量结果 {i} 有错误: {result['error']}")
            else:
                logger.info(f"✓ 批量结果 {i} 正常")
        
        return True
        
    except Exception as e:
        logger.error(f"批量推理测试失败: {e}")
        return False
    
    finally:
        # 清理测试文件
        for file_path in test_images + [model_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass


def main():
    """主测试函数"""
    logger.info("开始推理功能测试...")
    logger.info("=" * 60)
    
    test_results = []
    
    # 1. 测试推理类基本功能
    logger.info("1. 测试推理类基本功能")
    result1 = test_inference_class()
    test_results.append(("推理类基本功能", result1))
    logger.info("=" * 60)
    
    # 2. 测试真实数据
    logger.info("2. 测试真实数据")
    result2 = test_real_data()
    test_results.append(("真实数据测试", result2))
    logger.info("=" * 60)
    
    # 3. 测试批量推理
    logger.info("3. 测试批量推理")
    result3 = test_batch_inference()
    test_results.append(("批量推理测试", result3))
    logger.info("=" * 60)
    
    # 总结
    logger.info("测试总结:")
    all_passed = True
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 所有测试通过！推理功能正常。")
        return True
    else:
        logger.warning("⚠️  部分测试失败，请检查配置。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

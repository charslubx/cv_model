#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片推理脚本
支持读取图片并输出对应的类型分类结果

功能：
1. 加载训练好的模型
2. 读取和预处理图片
3. 进行推理预测
4. 输出各种类型的分类结果

作者: AI Assistant
日期: 2025-10-23
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from base_model import FullModel
from training import TextileNetDataset

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FashionInference:
    """服装图片推理类"""
    
    # DeepFashion属性标签映射
    DEEPFASHION_ATTRIBUTES = {
        0: "floral",           # 花卉图案
        1: "graphic",          # 图形图案  
        2: "striped",          # 条纹
        3: "embroidered",      # 刺绣
        4: "pleated",          # 褶皱
        5: "solid",            # 纯色
        6: "lattice",          # 格子
        7: "long_sleeve",      # 长袖
        8: "short_sleeve",     # 短袖
        9: "sleeveless",       # 无袖
        10: "maxi_length",     # 长款
        11: "mini_length",     # 短款
        12: "no_dress",        # 非连衣裙
        13: "crew_neckline",   # 圆领
        14: "v_neckline",      # V领
        15: "square_neckline", # 方领
        16: "no_neckline",     # 无领
        17: "denim",           # 牛仔
        18: "chiffon",         # 雪纺
        19: "cotton",          # 棉质
        20: "leather",         # 皮革
        21: "faux",            # 人造
        22: "knit",            # 针织
        23: "tight",           # 紧身
        24: "loose",           # 宽松
        25: "conventional"     # 常规
    }
    
    # 属性类别映射
    ATTRIBUTE_CATEGORIES = {
        "pattern": [0, 1, 2, 3, 4, 5, 6],        # 图案类型
        "sleeve": [7, 8, 9],                      # 袖子类型
        "length": [10, 11, 12],                   # 长度类型
        "neckline": [13, 14, 15, 16],            # 领口类型
        "material": [17, 18, 19, 20, 21, 22],    # 材质类型
        "fit": [23, 24, 25]                       # 版型类型
    }
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """初始化推理器
        
        Args:
            model_path: 训练好的模型路径
            device: 推理设备
        """
        self.device = device
        self.model = None
        self.transform = None
        
        # 类别映射字典
        self.fabric_classes = []
        self.fiber_classes = []
        self.deepfashion_attrs = []
        
        # 加载模型
        self.load_model(model_path)
        
        # 设置图片预处理
        self.setup_transform()
        
        # 加载类别信息
        self.load_class_info()
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            logger.info(f"正在加载模型: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            # 加载完整模型
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            
            logger.info("✓ 模型加载成功")
            logger.info(f"  设备: {self.device}")
            logger.info(f"  参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def setup_transform(self):
        """设置图片预处理"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        logger.info("✓ 图片预处理设置完成")
    
    def load_class_info(self):
        """加载类别信息"""
        try:
            # 尝试从数据集中获取类别信息
            textile_root = "/home/cv_model"
            
            # 加载Fabric类别
            if os.path.exists(os.path.join(textile_root, "fabric", "train")):
                try:
                    fabric_dataset = TextileNetDataset(
                        root_dir=textile_root,
                        dataset_type='fabric',
                        split='train',
                        transform=None
                    )
                    self.fabric_classes = fabric_dataset.get_class_names()
                    logger.info(f"✓ 加载Fabric类别: {len(self.fabric_classes)}个")
                except Exception as e:
                    logger.warning(f"无法加载Fabric类别: {e}")
                    self.fabric_classes = [f"fabric_class_{i}" for i in range(20)]
            
            # 加载Fiber类别
            if os.path.exists(os.path.join(textile_root, "fiber", "train")):
                try:
                    fiber_dataset = TextileNetDataset(
                        root_dir=textile_root,
                        dataset_type='fiber',
                        split='train',
                        transform=None
                    )
                    self.fiber_classes = fiber_dataset.get_class_names()
                    logger.info(f"✓ 加载Fiber类别: {len(self.fiber_classes)}个")
                except Exception as e:
                    logger.warning(f"无法加载Fiber类别: {e}")
                    self.fiber_classes = [f"fiber_class_{i}" for i in range(32)]
            
            # DeepFashion属性（示例）
            self.deepfashion_attrs = [
                "texture_1", "texture_2", "texture_3", "texture_4", "texture_5",
                "fabric_1", "fabric_2", "fabric_3", "fabric_4", "fabric_5",
                "shape_1", "shape_2", "shape_3", "shape_4", "shape_5",
                "part_1", "part_2", "part_3", "part_4", "part_5",
                "style_1", "style_2", "style_3", "style_4", "style_5",
                "fit_1"
            ]
            
            logger.info(f"✓ 类别信息加载完成")
            logger.info(f"  DeepFashion属性: {len(self.deepfashion_attrs)}个")
            logger.info(f"  Fabric类别: {len(self.fabric_classes)}个")
            logger.info(f"  Fiber类别: {len(self.fiber_classes)}个")
            
        except Exception as e:
            logger.warning(f"类别信息加载失败: {e}")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """加载和预处理图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            预处理后的图片张量
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
            
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            logger.info(f"✓ 图片加载成功: {image_path}")
            logger.info(f"  原始尺寸: {image.size}")
            
            # 预处理
            if self.transform:
                image_tensor = self.transform(image)
                # 添加batch维度
                image_tensor = image_tensor.unsqueeze(0)
                logger.info(f"  预处理后形状: {image_tensor.shape}")
                return image_tensor.to(self.device)
            else:
                raise ValueError("图片预处理器未初始化")
                
        except Exception as e:
            logger.error(f"图片加载失败: {e}")
            raise
    
    def predict(self, image_input: Union[str, torch.Tensor]) -> Dict:
        """进行推理预测
        
        Args:
            image_input: 图片路径或预处理后的张量
            
        Returns:
            包含各种预测结果的字典
        """
        try:
            # 处理输入
            if isinstance(image_input, str):
                image_tensor = self.load_image(image_input)
                image_path = image_input
            elif isinstance(image_input, torch.Tensor):
                image_tensor = image_input.to(self.device)
                image_path = "tensor_input"
            else:
                raise ValueError("不支持的输入类型")
            
            logger.info(f"开始推理: {image_path}")
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # 解析输出
            results = self.parse_outputs(outputs)
            
            logger.info("✓ 推理完成")
            return results
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def parse_outputs(self, outputs: Dict[str, torch.Tensor]) -> Dict:
        """解析模型输出
        
        Args:
            outputs: 模型原始输出
            
        Returns:
            解析后的预测结果
        """
        results = {
            'raw_outputs': {},
            'predictions': {},
            'probabilities': {},
            'top_predictions': {}
        }
        
        # 1. DeepFashion属性预测
        if 'attr_logits' in outputs:
            attr_logits = outputs['attr_logits']
            attr_probs = torch.sigmoid(attr_logits)
            
            results['raw_outputs']['deepfashion_attrs'] = attr_logits.cpu()
            results['probabilities']['deepfashion_attrs'] = attr_probs.cpu()
            
            # 获取高置信度的属性
            threshold = 0.5
            predicted_attrs = (attr_probs > threshold).cpu().numpy()[0]
            
            active_attrs = []
            for i, is_active in enumerate(predicted_attrs):
                if is_active and i < len(self.deepfashion_attrs):
                    confidence = attr_probs[0, i].item()
                    active_attrs.append({
                        'attribute': self.deepfashion_attrs[i],
                        'confidence': confidence
                    })
            
            # 按置信度排序
            active_attrs.sort(key=lambda x: x['confidence'], reverse=True)
            results['predictions']['deepfashion_attrs'] = active_attrs
            
            logger.info(f"  DeepFashion属性: {len(active_attrs)}个激活")
        
        # 2. Fabric纹理预测
        if 'fabric_logits' in outputs:
            fabric_logits = outputs['fabric_logits']
            fabric_probs = F.softmax(fabric_logits, dim=1)
            
            results['raw_outputs']['fabric'] = fabric_logits.cpu()
            results['probabilities']['fabric'] = fabric_probs.cpu()
            
            # 获取top-k预测
            top_k = min(5, len(self.fabric_classes))
            top_probs, top_indices = torch.topk(fabric_probs, top_k, dim=1)
            
            fabric_predictions = []
            for i in range(top_k):
                idx = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                if idx < len(self.fabric_classes):
                    fabric_predictions.append({
                        'class': self.fabric_classes[idx],
                        'confidence': prob
                    })
            
            results['predictions']['fabric'] = fabric_predictions[0] if fabric_predictions else None
            results['top_predictions']['fabric'] = fabric_predictions
            
            logger.info(f"  Fabric预测: {fabric_predictions[0]['class'] if fabric_predictions else 'None'}")
        
        # 3. Fiber纤维预测
        if 'fiber_logits' in outputs:
            fiber_logits = outputs['fiber_logits']
            fiber_probs = F.softmax(fiber_logits, dim=1)
            
            results['raw_outputs']['fiber'] = fiber_logits.cpu()
            results['probabilities']['fiber'] = fiber_probs.cpu()
            
            # 获取top-k预测
            top_k = min(5, len(self.fiber_classes))
            top_probs, top_indices = torch.topk(fiber_probs, top_k, dim=1)
            
            fiber_predictions = []
            for i in range(top_k):
                idx = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                if idx < len(self.fiber_classes):
                    fiber_predictions.append({
                        'class': self.fiber_classes[idx],
                        'confidence': prob
                    })
            
            results['predictions']['fiber'] = fiber_predictions[0] if fiber_predictions else None
            results['top_predictions']['fiber'] = fiber_predictions
            
            logger.info(f"  Fiber预测: {fiber_predictions[0]['class'] if fiber_predictions else 'None'}")
        
        # 4. 统一纹理预测
        if 'textile_logits' in outputs:
            textile_logits = outputs['textile_logits']
            textile_probs = F.softmax(textile_logits, dim=1)
            
            results['raw_outputs']['textile'] = textile_logits.cpu()
            results['probabilities']['textile'] = textile_probs.cpu()
            
            # 获取最高置信度预测
            max_prob, max_idx = torch.max(textile_probs, dim=1)
            results['predictions']['textile'] = {
                'class_index': max_idx.item(),
                'confidence': max_prob.item()
            }
            
            logger.info(f"  Textile预测: 类别{max_idx.item()}, 置信度{max_prob.item():.3f}")
        
        # 5. 分割预测（如果有）
        if 'seg_logits' in outputs:
            seg_logits = outputs['seg_logits']
            seg_probs = torch.sigmoid(seg_logits)
            
            results['raw_outputs']['segmentation'] = seg_logits.cpu()
            results['probabilities']['segmentation'] = seg_probs.cpu()
            
            # 生成分割掩码
            seg_mask = (seg_probs > 0.5).cpu().numpy()[0, 0]
            results['predictions']['segmentation'] = {
                'mask': seg_mask,
                'coverage': seg_mask.mean()
            }
            
            logger.info(f"  分割预测: 覆盖率{seg_mask.mean():.3f}")
        
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """批量预测
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            预测结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"批量预测失败 {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def format_results(self, results: Dict, detailed: bool = False) -> str:
        """格式化输出结果
        
        Args:
            results: 预测结果字典
            detailed: 是否显示详细信息
            
        Returns:
            格式化的结果字符串
        """
        output = []
        output.append("=" * 60)
        output.append("图片分类结果")
        output.append("=" * 60)
        
        # DeepFashion属性 - 按类别组织显示
        if 'deepfashion_attrs' in results['predictions']:
            attrs = results['predictions']['deepfashion_attrs']
            output.append(f"\n👗 服装属性分析:")
            
            # 按类别组织属性
            categorized_attrs = {}
            for attr in attrs:
                attr_name = attr['attribute']
                confidence = attr['confidence']
                
                # 找到属性对应的真实标签
                attr_idx = None
                for idx, name in self.DEEPFASHION_ATTRIBUTES.items():
                    if attr_name.endswith(f"_{idx}") or attr_name == name:
                        attr_idx = idx
                        break
                
                if attr_idx is not None:
                    real_name = self.DEEPFASHION_ATTRIBUTES[attr_idx]
                    
                    # 找到属性类别
                    category = None
                    for cat_name, indices in self.ATTRIBUTE_CATEGORIES.items():
                        if attr_idx in indices:
                            category = cat_name
                            break
                    
                    if category:
                        if category not in categorized_attrs:
                            categorized_attrs[category] = []
                        categorized_attrs[category].append({
                            'name': real_name,
                            'confidence': confidence
                        })
            
            # 显示各类别的属性
            category_names = {
                'pattern': '🎨 图案',
                'sleeve': '👕 袖型', 
                'length': '📏 长度',
                'neckline': '👔 领型',
                'material': '🧵 材质',
                'fit': '📐 版型'
            }
            
            for category, attrs_list in categorized_attrs.items():
                if attrs_list:
                    cat_name = category_names.get(category, category)
                    output.append(f"\n  {cat_name}:")
                    # 按置信度排序，显示前3个
                    sorted_attrs = sorted(attrs_list, key=lambda x: x['confidence'], reverse=True)
                    for attr in sorted_attrs[:3]:
                        output.append(f"    • {attr['name']}: {attr['confidence']:.3f}")
            
            if not categorized_attrs:
                # 如果无法分类，显示原始结果
                output.append("  检测到的属性:")
                for attr in attrs[:5]:
                    output.append(f"    • {attr['attribute']}: {attr['confidence']:.3f}")
        
        # Fabric预测
        if 'fabric' in results['predictions'] and results['predictions']['fabric']:
            fabric = results['predictions']['fabric']
            fabric_name = fabric['class']
            confidence = fabric['confidence']
            
            # 添加面料类型的中文说明
            fabric_translations = {
                'lace': '蕾丝',
                'denim': '牛仔布',
                'cotton': '棉布',
                'silk': '丝绸',
                'wool': '羊毛',
                'polyester': '聚酯纤维',
                'leather': '皮革',
                'chiffon': '雪纺',
                'knit': '针织物'
            }
            
            chinese_name = fabric_translations.get(fabric_name, fabric_name)
            output.append(f"\n🧵 面料类型:")
            if chinese_name != fabric_name:
                output.append(f"  • {chinese_name} ({fabric_name}): {confidence:.3f}")
            else:
                output.append(f"  • {fabric_name}: {confidence:.3f}")
            
            if detailed and 'fabric' in results['top_predictions']:
                output.append("  其他可能的面料:")
                for pred in results['top_predictions']['fabric'][:5]:
                    pred_chinese = fabric_translations.get(pred['class'], pred['class'])
                    if pred_chinese != pred['class']:
                        output.append(f"    - {pred_chinese} ({pred['class']}): {pred['confidence']:.3f}")
                    else:
                        output.append(f"    - {pred['class']}: {pred['confidence']:.3f}")
        
        # Fiber预测
        if 'fiber' in results['predictions'] and results['predictions']['fiber']:
            fiber = results['predictions']['fiber']
            fiber_name = fiber['class']
            confidence = fiber['confidence']
            
            # 添加纤维类型的中文说明
            fiber_translations = {
                'cotton': '棉纤维',
                'wool': '羊毛纤维',
                'silk': '丝纤维',
                'polyester': '聚酯纤维',
                'nylon': '尼龙纤维',
                'acrylic': '腈纶纤维',
                'linen': '亚麻纤维',
                'rayon': '人造丝',
                'llama': '羊驼毛'
            }
            
            chinese_name = fiber_translations.get(fiber_name, fiber_name)
            output.append(f"\n🧶 纤维类型:")
            if chinese_name != fiber_name:
                output.append(f"  • {chinese_name} ({fiber_name}): {confidence:.3f}")
            else:
                output.append(f"  • {fiber_name}: {confidence:.3f}")
            
            if detailed and 'fiber' in results['top_predictions']:
                output.append("  其他可能的纤维:")
                for pred in results['top_predictions']['fiber'][:5]:
                    pred_chinese = fiber_translations.get(pred['class'], pred['class'])
                    if pred_chinese != pred['class']:
                        output.append(f"    - {pred_chinese} ({pred['class']}): {pred['confidence']:.3f}")
                    else:
                        output.append(f"    - {pred['class']}: {pred['confidence']:.3f}")
        
        # 分割结果
        if 'segmentation' in results['predictions']:
            seg = results['predictions']['segmentation']
            output.append(f"\n✂️ 分割结果:")
            output.append(f"  • 覆盖率: {seg['coverage']:.3f}")
        
        # 智能总结
        output.append(f"\n🎯 智能分析总结:")
        summary_parts = []
        
        # 从属性中提取关键信息
        if 'deepfashion_attrs' in results['predictions']:
            attrs = results['predictions']['deepfashion_attrs']
            if attrs:
                # 找到最高置信度的属性
                top_attr = max(attrs, key=lambda x: x['confidence'])
                attr_name = top_attr['attribute']
                
                # 尝试解析属性名
                for idx, name in self.DEEPFASHION_ATTRIBUTES.items():
                    if attr_name.endswith(f"_{idx}"):
                        summary_parts.append(f"主要特征为{name}")
                        break
        
        # 添加面料信息
        if 'fabric' in results['predictions'] and results['predictions']['fabric']:
            fabric = results['predictions']['fabric']
            fabric_translations = {
                'lace': '蕾丝', 'denim': '牛仔布', 'cotton': '棉布', 'silk': '丝绸',
                'wool': '羊毛', 'polyester': '聚酯纤维', 'leather': '皮革', 'chiffon': '雪纺'
            }
            fabric_chinese = fabric_translations.get(fabric['class'], fabric['class'])
            summary_parts.append(f"面料为{fabric_chinese}")
        
        # 添加纤维信息
        if 'fiber' in results['predictions'] and results['predictions']['fiber']:
            fiber = results['predictions']['fiber']
            fiber_translations = {
                'cotton': '棉纤维', 'wool': '羊毛纤维', 'silk': '丝纤维',
                'polyester': '聚酯纤维', 'llama': '羊驼毛'
            }
            fiber_chinese = fiber_translations.get(fiber['class'], fiber['class'])
            summary_parts.append(f"纤维为{fiber_chinese}")
        
        if summary_parts:
            output.append(f"  这是一件{', '.join(summary_parts)}的服装。")
        else:
            output.append("  未能识别出明确的服装特征。")
        
        output.append("=" * 60)
        return "\n".join(output)


def main():
    """主函数 - 演示推理功能"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='服装图片推理脚本')
    parser.add_argument('--model_path', type=str, 
                       default="smart_mixed_checkpoints/best_model.pth",
                       help='训练好的模型路径')
    parser.add_argument('--image_path', type=str,
                       help='要推理的图片路径')
    parser.add_argument('--output_detail', action='store_true',
                       help='输出详细的预测结果')
    
    args = parser.parse_args()
    
    logger.info("开始图片推理演示...")
    
    # 配置
    model_path = args.model_path
    test_images = []
    
    # 如果指定了图片路径，添加到测试列表
    if args.image_path:
        test_images.append(args.image_path)
    else:
        # 默认测试图片（如果没有指定）
        test_images = [
            # 可以添加测试图片路径
            # "/path/to/test/image1.jpg",
            # "/path/to/test/image2.jpg",
        ]
    
    try:
        # 检查模型文件
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            logger.info("请先训练模型或提供正确的模型路径")
            return
        
        # 创建推理器
        inferencer = FashionInference(model_path)
        
        # 如果没有指定测试图片，尝试从数据集中找一些
        if not test_images:
            logger.info("正在寻找测试图片...")
            
            # 从fabric数据集中找一些图片
            fabric_dir = "/home/cv_model/fabric/train"
            if os.path.exists(fabric_dir):
                for class_dir in os.listdir(fabric_dir)[:3]:  # 取前3个类别
                    class_path = os.path.join(fabric_dir, class_dir)
                    if os.path.isdir(class_path):
                        images = [f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            test_images.append(os.path.join(class_path, images[0]))
            
            # 从fiber数据集中找一些图片
            fiber_dir = "/home/cv_model/fiber/train"
            if os.path.exists(fiber_dir):
                for class_dir in os.listdir(fiber_dir)[:2]:  # 取前2个类别
                    class_path = os.path.join(fiber_dir, class_dir)
                    if os.path.isdir(class_path):
                        images = [f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        if images:
                            test_images.append(os.path.join(class_path, images[0]))
        
        if not test_images:
            logger.warning("没有找到测试图片")
            logger.info("请在代码中指定测试图片路径或确保数据集存在")
            return
        
        logger.info(f"找到 {len(test_images)} 张测试图片")
        
        # 进行推理
        for i, image_path in enumerate(test_images):
            logger.info(f"\n处理图片 {i+1}/{len(test_images)}: {image_path}")
            
            try:
                # 检查图片文件是否存在
                if not os.path.exists(image_path):
                    logger.error(f"图片文件不存在: {image_path}")
                    continue
                
                # 单张图片推理
                results = inferencer.predict(image_path)
                
                # 格式化并显示结果
                detailed = args.output_detail if 'args' in locals() else True
                formatted_results = inferencer.format_results(results, detailed=detailed)
                print(formatted_results)
                
            except Exception as e:
                logger.error(f"图片 {image_path} 推理失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n✓ 推理演示完成")
        
    except Exception as e:
        logger.error(f"推理演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

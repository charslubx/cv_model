"""
推理模块 - 封装完整的图片预处理、模型推理和结果后处理流程
支持单张图片和批量图片推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from typing import Union, List, Dict, Optional
import logging

# 导入数据集相关的定义和函数
try:
    from base_model import ATTR_DEFS, CATEGORY_DEFS, read_attr_cloth_file
except ImportError:
    ATTR_DEFS = None
    CATEGORY_DEFS = None
    logger.warning("无法从base_model导入属性定义，将使用手动指定的属性名称")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FashionInferenceWrapper:
    """服装属性推理包装器"""
    
    def __init__(
        self,
        model: nn.Module,
        attr_names: Optional[List[str]] = None,
        fabric_names: Optional[List[str]] = None,
        fiber_names: Optional[List[str]] = None,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
        img_size: int = 224,
        enable_textile_classification: bool = False
    ):
        """
        初始化推理包装器
        
        Args:
            model: 训练好的模型
            attr_names: 属性名称列表（按索引顺序）
            fabric_names: Fabric类别名称列表（可选）
            fiber_names: Fiber类别名称列表（可选）
            threshold: 属性判定阈值，默认0.5
            device: 运行设备，默认自动选择
            img_size: 输入图片尺寸
            enable_textile_classification: 是否启用纹理分类
        """
        self.model = model
        self.model.eval()  # 设置为评估模式
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 类别名称
        self.attr_names = attr_names or self._get_default_attr_names()
        self.fabric_names = fabric_names
        self.fiber_names = fiber_names
        self.threshold = threshold
        self.enable_textile_classification = enable_textile_classification
        
        # 图片预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info(f"推理包装器初始化完成")
        logger.info(f"- 运行设备: {self.device}")
        logger.info(f"- 属性数量: {len(self.attr_names)}")
        logger.info(f"- 判定阈值: {self.threshold}")
        logger.info(f"- 纹理分类: {'启用' if enable_textile_classification else '禁用'}")
    
    def _get_default_attr_names(self) -> List[str]:
        """获取默认属性名称列表（从数据集定义中读取）"""
        # 优先从base_model中导入的全局变量ATTR_DEFS获取
        if ATTR_DEFS is not None and len(ATTR_DEFS) > 0:
            attr_names = [attr['name'] for attr in ATTR_DEFS]
            logger.info(f"从ATTR_DEFS加载了 {len(attr_names)} 个属性")
            return attr_names
        
        # 如果导入失败，尝试直接读取文件
        default_attr_file = "/home/cv_model/deepfashion/Category and Attribute Prediction Benchmark/Anno_fine/list_attr_cloth.txt"
        if os.path.exists(default_attr_file):
            attr_names = load_attr_names_from_file(default_attr_file)
            if attr_names:
                logger.info(f"从文件 {default_attr_file} 加载了 {len(attr_names)} 个属性")
                return attr_names
        
        # 如果都失败了，返回警告并使用空列表
        logger.warning("无法自动加载属性名称，请手动指定 attr_names 参数！")
        logger.warning("使用方法: wrapper = FashionInferenceWrapper(model=model, attr_names=your_attr_list)")
        return []
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            预处理后的图片tensor [1, 3, H, W]
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0)  # [1, 3, H, W]
            return img_tensor.to(self.device)
        except Exception as e:
            logger.error(f"加载图片失败 {image_path}: {str(e)}")
            raise
    
    def load_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        加载并预处理多张图片
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            预处理后的图片tensor [N, 3, H, W]
        """
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.preprocess(img)
                images.append(img_tensor)
            except Exception as e:
                logger.error(f"加载图片失败 {path}: {str(e)}")
                raise
        
        return torch.stack(images).to(self.device)
    
    def predict_single(
        self,
        image_path: str,
        return_raw: bool = False
    ) -> Dict:
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            return_raw: 是否返回原始logits和概率，默认False
            
        Returns:
            预测结果字典
        """
        # 加载图片
        img_tensor = self.load_image(image_path)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # 后处理
        result = self._process_outputs(outputs, 0, return_raw)
        result['image_path'] = image_path
        result['image_name'] = os.path.basename(image_path)
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        return_raw: bool = False
    ) -> List[Dict]:
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
            return_raw: 是否返回原始logits和概率
            
        Returns:
            预测结果列表
        """
        # 加载图片
        img_tensors = self.load_images(image_paths)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(img_tensors)
        
        # 批量后处理
        results = []
        for i, image_path in enumerate(image_paths):
            result = self._process_outputs(outputs, i, return_raw)
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            results.append(result)
        
        return results
    
    def _process_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        index: int,
        return_raw: bool = False
    ) -> Dict:
        """
        处理模型输出，转换为可读的结果
        
        Args:
            outputs: 模型输出字典
            index: 批次中的索引
            return_raw: 是否返回原始数据
            
        Returns:
            处理后的结果字典
        """
        result = {}
        
        # 1. 处理属性分类
        if 'attr_logits' in outputs:
            attr_logits = outputs['attr_logits'][index]  # [num_classes]
            attr_probs = torch.sigmoid(attr_logits).cpu().numpy()
            
            # 判定存在的属性
            predicted_indices = np.where(attr_probs > self.threshold)[0]
            predicted_attrs = [self.attr_names[i] for i in predicted_indices]
            
            # 构建置信度字典（只包含预测为正的属性）
            confidence_scores = {
                self.attr_names[i]: float(attr_probs[i])
                for i in predicted_indices
            }
            
            # 排序（按置信度降序）
            confidence_scores = dict(
                sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
            )
            
            result['attributes'] = {
                'predicted': predicted_attrs,
                'count': len(predicted_attrs),
                'confidence_scores': confidence_scores
            }
            
            # 返回所有属性的置信度（可选）
            if return_raw:
                result['attributes']['all_scores'] = {
                    self.attr_names[i]: float(attr_probs[i])
                    for i in range(len(self.attr_names))
                }
        
        # 2. 处理Fabric纹理分类
        if self.enable_textile_classification and 'fabric_logits' in outputs:
            fabric_logits = outputs['fabric_logits'][index]
            fabric_probs = torch.softmax(fabric_logits, dim=0).cpu().numpy()
            fabric_class = int(np.argmax(fabric_probs))
            fabric_confidence = float(fabric_probs[fabric_class])
            
            result['fabric'] = {
                'class_id': fabric_class,
                'confidence': fabric_confidence
            }
            
            # 如果提供了类别名称，添加类别名
            if self.fabric_names and fabric_class < len(self.fabric_names):
                result['fabric']['class_name'] = self.fabric_names[fabric_class]
            
            # 返回所有类别的概率（可选）
            if return_raw and self.fabric_names:
                result['fabric']['all_probs'] = {
                    self.fabric_names[i]: float(fabric_probs[i])
                    for i in range(min(len(self.fabric_names), len(fabric_probs)))
                }
        
        # 3. 处理Fiber纤维分类
        if self.enable_textile_classification and 'fiber_logits' in outputs:
            fiber_logits = outputs['fiber_logits'][index]
            fiber_probs = torch.softmax(fiber_logits, dim=0).cpu().numpy()
            fiber_class = int(np.argmax(fiber_probs))
            fiber_confidence = float(fiber_probs[fiber_class])
            
            result['fiber'] = {
                'class_id': fiber_class,
                'confidence': fiber_confidence
            }
            
            # 如果提供了类别名称，添加类别名
            if self.fiber_names and fiber_class < len(self.fiber_names):
                result['fiber']['class_name'] = self.fiber_names[fiber_class]
            
            # 返回所有类别的概率（可选）
            if return_raw and self.fiber_names:
                result['fiber']['all_probs'] = {
                    self.fiber_names[i]: float(fiber_probs[i])
                    for i in range(min(len(self.fiber_names), len(fiber_probs)))
                }
        
        # 4. 处理分割（如果有）
        if 'seg_logits' in outputs:
            seg_logits = outputs['seg_logits'][index]  # [1, H, W]
            seg_mask = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy().astype(np.uint8)
            
            result['segmentation'] = {
                'has_mask': True,
                'mask_shape': seg_mask.shape,
                'coverage_ratio': float(seg_mask.mean())  # 分割区域占比
            }
            
            if return_raw:
                result['segmentation']['mask'] = seg_mask.tolist()
        
        # 5. 类别权重（如果有）
        if 'class_weights' in outputs and return_raw:
            class_weights = outputs['class_weights'][index].cpu().numpy()
            result['class_weights'] = {
                self.attr_names[i]: float(class_weights[i])
                for i in range(len(self.attr_names))
            }
        
        return result
    
    def predict_and_save(
        self,
        image_path: str,
        output_path: str,
        return_raw: bool = False
    ):
        """
        预测并保存结果到JSON文件
        
        Args:
            image_path: 图片路径
            output_path: 输出JSON文件路径
            return_raw: 是否包含原始数据
        """
        result = self.predict_single(image_path, return_raw)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"预测结果已保存到: {output_path}")
        return result
    
    def predict_batch_and_save(
        self,
        image_paths: List[str],
        output_path: str,
        return_raw: bool = False
    ):
        """
        批量预测并保存结果
        
        Args:
            image_paths: 图片路径列表
            output_path: 输出JSON文件路径
            return_raw: 是否包含原始数据
        """
        results = self.predict_batch(image_paths, return_raw)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"批量预测结果已保存到: {output_path}")
        return results
    
    def get_summary(self, result: Dict) -> str:
        """
        生成预测结果的可读摘要
        
        Args:
            result: 预测结果字典
            
        Returns:
            摘要文本
        """
        lines = []
        lines.append(f"图片: {result.get('image_name', 'Unknown')}")
        lines.append("-" * 50)
        
        # 属性
        if 'attributes' in result:
            attrs = result['attributes']
            lines.append(f"检测到的属性 ({attrs['count']}个):")
            for attr, conf in attrs['confidence_scores'].items():
                lines.append(f"  - {attr}: {conf:.2%}")
        
        # Fabric
        if 'fabric' in result:
            fabric = result['fabric']
            fabric_info = f"Fabric类别: {fabric.get('class_name', f'ID {fabric['class_id']}')}"
            fabric_info += f" (置信度: {fabric['confidence']:.2%})"
            lines.append(fabric_info)
        
        # Fiber
        if 'fiber' in result:
            fiber = result['fiber']
            fiber_info = f"Fiber类别: {fiber.get('class_name', f'ID {fiber['class_id']}')}"
            fiber_info += f" (置信度: {fiber['confidence']:.2%})"
            lines.append(fiber_info)
        
        # 分割
        if 'segmentation' in result:
            seg = result['segmentation']
            lines.append(f"分割: 覆盖率 {seg['coverage_ratio']:.2%}")
        
        return "\n".join(lines)


def load_attr_names_from_file(file_path: str) -> List[str]:
    """
    从DeepFashion属性定义文件加载属性名称
    
    Args:
        file_path: list_attr_cloth.txt 文件路径
        
    Returns:
        属性名称列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是属性数量
            num_attrs = int(lines[0].strip())
            # 第二行是表头
            # 从第三行开始是属性定义
            attr_names = []
            for line in lines[2:2 + num_attrs]:
                parts = line.strip().split()
                if len(parts) >= 1:
                    attr_name = parts[0]
                    attr_names.append(attr_name)
            logger.info(f"成功从 {file_path} 加载了 {len(attr_names)} 个属性")
            return attr_names
    except Exception as e:
        logger.error(f"读取属性定义文件失败 {file_path}: {str(e)}")
        return []


def get_attr_names_from_dataset(dataset_root: str = "/home/cv_model/deepfashion") -> List[str]:
    """
    从DeepFashion数据集根目录自动加载属性名称
    
    Args:
        dataset_root: DeepFashion数据集根目录
        
    Returns:
        属性名称列表
    """
    attr_file = os.path.join(
        dataset_root,
        "Category and Attribute Prediction Benchmark",
        "Anno_fine",
        "list_attr_cloth.txt"
    )
    
    if os.path.exists(attr_file):
        return load_attr_names_from_file(attr_file)
    else:
        logger.error(f"属性定义文件不存在: {attr_file}")
        logger.error(f"请检查数据集路径或手动指定属性名称列表")
        return []


def load_category_names_from_file(file_path: str) -> List[str]:
    """
    从DeepFashion类别定义文件加载类别名称
    
    Args:
        file_path: list_category_cloth.txt 文件路径
        
    Returns:
        类别名称列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是类别数量
            num_categories = int(lines[0].strip())
            # 第二行是表头
            # 从第三行开始是类别定义
            category_names = []
            for line in lines[2:2 + num_categories]:
                parts = line.strip().split()
                category_name = parts[0]
                category_names.append(category_name)
            return category_names
    except Exception as e:
        logger.error(f"读取类别定义文件失败: {str(e)}")
        return []


# 测试代码
if __name__ == "__main__":
    # 示例：如何使用推理包装器
    print("推理包装器模块已加载")
    print("使用示例请参考 inference_example.py")

"""
模型加载和推理模块
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_model import FullModel

logger = logging.getLogger(__name__)


class ModelInference:
    """模型推理类"""
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        num_classes: int = 26,
        img_size: int = 224
    ):
        """
        初始化模型推理类
        
        Args:
            model_path: 模型权重文件路径
            device: 推理设备 ("cuda" 或 "cpu")
            num_classes: 类别数量
            img_size: 输入图片大小
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 初始化模型
        logger.info(f"正在初始化模型，类别数: {num_classes}")
        self.model = FullModel(
            num_classes=num_classes,
            cnn_type='resnet50',
            weights='IMAGENET1K_V1',
            enable_segmentation=False,  # 推理时不需要分割
            gat_dims=[1024, 512],
            gat_heads=4,
            gat_dropout=0.2
        )
        
        # 加载模型权重
        if model_path and os.path.exists(model_path):
            logger.info(f"正在加载模型权重: {model_path}")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("模型权重加载成功")
            except Exception as e:
                logger.warning(f"加载模型权重失败: {str(e)}")
                logger.warning("使用预训练权重初始化")
        else:
            logger.warning(f"模型权重文件不存在: {model_path}")
            logger.warning("使用预训练权重初始化")
        
        # 将模型移到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 定义图片预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 属性名称映射 (这里使用默认的26个属性)
        self.attribute_names = self._get_attribute_names()
        
        logger.info(f"模型初始化完成，设备: {self.device}")
    
    def _get_attribute_names(self) -> List[str]:
        """
        获取属性名称列表
        默认使用DeepFashion数据集的26个属性
        """
        # 这里是DeepFashion数据集的属性名称
        # 可以根据实际情况修改
        return [
            "black", "blue", "brown", "collar", "cyan",
            "gray", "green", "many_colors", "necktie", "pattern_floral",
            "pattern_graphics", "pattern_plaid", "pattern_solid", "pattern_spot", "pattern_stripe",
            "pink", "purple", "red", "scarf", "skin_exposure",
            "white", "yellow", "denim", "knitted", "leather",
            "cotton"
        ]
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        图片预处理
        
        Args:
            image: PIL图片对象
            
        Returns:
            预处理后的张量
        """
        # 确保图片是RGB模式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 应用预处理
        img_tensor = self.transform(image)
        
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    @torch.no_grad()
    def predict(
        self, 
        image: Union[Image.Image, torch.Tensor],
        threshold: float = 0.5
    ) -> Dict:
        """
        单张图片分类（多标签分类）
        
        Args:
            image: PIL图片或张量
            threshold: 分类阈值，默认0.5
            
        Returns:
            分类结果字典，包含：
            - attributes: 所有属性的置信度
            - classifications: 所有属性的分类结果（0或1）
            - positive_attributes: 预测为正的属性列表
            - top_k_attributes: Top-K置信度最高的属性
        """
        # 预处理图片
        if isinstance(image, Image.Image):
            img_tensor = self.preprocess_image(image)
        else:
            img_tensor = image
        
        # 移到指定设备
        img_tensor = img_tensor.to(self.device)
        
        # 前向传播
        try:
            outputs = self.model(img_tensor)
            
            # 获取属性分类logits
            attr_logits = outputs['attr_logits']
            
            # 应用sigmoid获取概率（多标签分类）
            attr_probs = torch.sigmoid(attr_logits).cpu().numpy()[0]
            
            # 使用阈值进行二值化分类
            attr_classes = (attr_probs >= threshold).astype(int)
            
            # 构建结果字典
            predictions = {
                "attributes": {},           # 置信度
                "classifications": {},       # 分类结果（0或1）
                "positive_attributes": [],   # 预测为正的属性
                "top_k_attributes": []       # Top-K属性
            }
            
            # 添加所有属性的置信度和分类结果
            for attr_name, prob, cls in zip(
                self.attribute_names, 
                attr_probs, 
                attr_classes
            ):
                predictions["attributes"][attr_name] = float(prob)
                predictions["classifications"][attr_name] = int(cls)
                
                # 收集预测为正的属性
                if cls == 1:
                    predictions["positive_attributes"].append({
                        "attribute": attr_name,
                        "confidence": float(prob)
                    })
            
            # 按置信度排序正属性
            predictions["positive_attributes"].sort(
                key=lambda x: x["confidence"],
                reverse=True
            )
            
            # 获取Top-K置信度最高的属性
            top_k = 5
            top_indices = attr_probs.argsort()[-top_k:][::-1]
            for idx in top_indices:
                predictions["top_k_attributes"].append({
                    "attribute": self.attribute_names[idx],
                    "confidence": float(attr_probs[idx]),
                    "classification": int(attr_classes[idx])
                })
            
            # 添加统计信息
            predictions["statistics"] = {
                "total_attributes": len(self.attribute_names),
                "positive_count": int(attr_classes.sum()),
                "threshold": threshold
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"分类过程出错: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Image.Image],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        批量图片分类（多标签分类）
        
        Args:
            images: PIL图片列表
            threshold: 分类阈值，默认0.5
            
        Returns:
            分类结果列表
        """
        # 预处理所有图片
        img_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.cat(img_tensors, dim=0).to(self.device)
        
        # 前向传播
        try:
            outputs = self.model(batch_tensor)
            attr_logits = outputs['attr_logits']
            attr_probs = torch.sigmoid(attr_logits).cpu().numpy()
            
            # 使用阈值进行二值化分类
            attr_classes = (attr_probs >= threshold).astype(int)
            
            # 构建结果列表
            results = []
            for probs, classes in zip(attr_probs, attr_classes):
                predictions = {
                    "attributes": {},
                    "classifications": {},
                    "positive_attributes": [],
                    "top_k_attributes": []
                }
                
                # 添加所有属性的置信度和分类结果
                for attr_name, prob, cls in zip(
                    self.attribute_names,
                    probs,
                    classes
                ):
                    predictions["attributes"][attr_name] = float(prob)
                    predictions["classifications"][attr_name] = int(cls)
                    
                    # 收集预测为正的属性
                    if cls == 1:
                        predictions["positive_attributes"].append({
                            "attribute": attr_name,
                            "confidence": float(prob)
                        })
                
                # 按置信度排序正属性
                predictions["positive_attributes"].sort(
                    key=lambda x: x["confidence"],
                    reverse=True
                )
                
                # 获取Top-K属性
                top_k = 5
                top_indices = probs.argsort()[-top_k:][::-1]
                for idx in top_indices:
                    predictions["top_k_attributes"].append({
                        "attribute": self.attribute_names[idx],
                        "confidence": float(probs[idx]),
                        "classification": int(classes[idx])
                    })
                
                # 添加统计信息
                predictions["statistics"] = {
                    "total_attributes": len(self.attribute_names),
                    "positive_count": int(classes.sum()),
                    "threshold": threshold
                }
                
                results.append(predictions)
            
            return results
            
        except Exception as e:
            logger.error(f"批量分类过程出错: {str(e)}")
            raise
    
    def get_attribute_names(self) -> List[str]:
        """获取属性名称列表"""
        return self.attribute_names

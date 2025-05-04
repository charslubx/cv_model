import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import logging
import random

logger = logging.getLogger(__name__)

class DeepFashionDataset(Dataset):
    """DeepFashion多任务数据集"""

    def __init__(self,
                 img_list_file: str,  # 图片列表文件
                 attr_file: str,  # 属性标注文件
                 cate_file: str = None,  # 类别标注文件
                 bbox_file: str = None,  # 边界框标注文件
                 seg_file: str = None,  # 分割标注文件
                 image_dir: str = None,  # 图像根目录
                 transform=None,
                 max_retry: int = 5):
        """初始化数据集

        Args:
            img_list_file: 图片列表文件路径
            attr_file: 属性标注文件路径
            cate_file: 类别标注文件路径（可选）
            bbox_file: 边界框标注文件路径（可选）
            seg_file: 分割标注文件路径（可选）
            image_dir: 图像根目录路径（可选）
            transform: 数据增强和预处理
            max_retry: 加载失败时的最大重试次数
        """
        self.transform = transform
        self.image_dir = image_dir
        self.max_retry = max_retry

        # 读取图片列表
        try:
            with open(img_list_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    raise ValueError(f"图片列表文件为空: {img_list_file}")
                self.img_paths = [line.strip() for line in lines]
            logger.info(f"读取了 {len(self.img_paths)} 个图片路径")
        except Exception as e:
            logger.error(f"读取图片列表文件失败: {str(e)}")
            raise

        # 读取属性标注
        try:
            with open(attr_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) < 1:
                    raise ValueError(f"属性标注文件为空: {attr_file}")
                    
                # 将属性值从1/-1转换为1/0
                self.attr_labels = []
                for line in lines:
                    attrs = [(int(x) + 1) // 2 for x in line.strip().split()]
                    self.attr_labels.append(attrs)
                
                self.attr_labels = torch.tensor(self.attr_labels, dtype=torch.float32)
                logger.info(f"读取了 {len(self.attr_labels)} 个属性标注")
                
                # 验证图片路径和属性标注数量是否匹配
                if len(self.img_paths) != len(self.attr_labels):
                    raise ValueError(f"图片数量({len(self.img_paths)})与属性标注数量({len(self.attr_labels)})不匹配")
        except Exception as e:
            logger.error(f"读取属性标注文件失败: {str(e)}")
            raise

        # 读取类别标注（如果有）
        self.cate_labels = None
        if cate_file and os.path.exists(cate_file):
            try:
                with open(cate_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    categories = [int(line.strip()) for line in lines]
                    self.cate_labels = torch.tensor(categories, dtype=torch.long)
                    logger.info(f"读取了 {len(self.cate_labels)} 个类别标注")
                    
                    # 验证数量是否匹配
                    if len(self.img_paths) != len(self.cate_labels):
                        raise ValueError(f"图片数量({len(self.img_paths)})与类别标注数量({len(self.cate_labels)})不匹配")
            except Exception as e:
                logger.error(f"读取类别标注文件失败: {str(e)}")

        # 读取边界框标注（如果有）
        self.bbox_labels = None
        if bbox_file and os.path.exists(bbox_file):
            try:
                with open(bbox_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    bboxes = []
                    for line in lines:
                        bbox = [float(x) for x in line.strip().split()]
                        bboxes.append(bbox)
                    self.bbox_labels = torch.tensor(bboxes, dtype=torch.float32)
                    logger.info(f"读取了 {len(self.bbox_labels)} 个边界框标注")
                    
                    # 验证数量是否匹配
                    if len(self.img_paths) != len(self.bbox_labels):
                        raise ValueError(f"图片数量({len(self.img_paths)})与边界框标注数量({len(self.bbox_labels)})不匹配")
            except Exception as e:
                logger.error(f"读取边界框标注文件失败: {str(e)}")

        # 读取分割标注（如果有）
        self.seg_labels = None
        if seg_file and os.path.exists(seg_file):
            try:
                with open(seg_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    segmentations = []
                    for line in lines:
                        seg = [float(x) for x in line.strip().split()]
                        segmentations.append(seg)
                    self.seg_labels = torch.tensor(segmentations, dtype=torch.float32)
                    logger.info(f"读取了 {len(self.seg_labels)} 个分割标注")
                    
                    # 验证数量是否匹配
                    if len(self.img_paths) != len(self.seg_labels):
                        raise ValueError(f"图片数量({len(self.img_paths)})与分割标注数量({len(self.seg_labels)})不匹配")
            except Exception as e:
                logger.error(f"读取分割标注文件失败: {str(e)}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """获取单个样本"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                # 获取图片路径
                img_path = self.img_paths[idx]
                if self.image_dir:
                    img_path = os.path.join(self.image_dir, img_path)

                # 读取图片
                image = Image.open(img_path).convert('RGB')

                # 数据增强
                if self.transform:
                    image = self.transform(image)

                # 准备样本数据
                sample = {
                    'image': image,
                    'img_name': self.img_paths[idx],  # 保存相对路径
                    'attr_labels': self.attr_labels[idx]
                }

                # 添加类别标签（如果有）
                if self.cate_labels is not None:
                    sample['category_labels'] = self.cate_labels[idx]

                # 添加边界框标签（如果有）
                if self.bbox_labels is not None:
                    sample['bbox_labels'] = self.bbox_labels[idx]

                # 添加分割标签（如果有）
                if self.seg_labels is not None:
                    sample['segmentation'] = self.seg_labels[idx]

                return sample

            except Exception as e:
                retry_count += 1
                logger.warning(f"加载样本 {idx} 失败 (尝试 {retry_count}/{self.max_retry}): {str(e)}")
                if retry_count == self.max_retry:
                    logger.error(f"无法加载样本 {idx}，已达到最大重试次数")
                    # 返回一个空样本
                    return {
                        'image': torch.zeros((3, 224, 224)),
                        'img_name': self.img_paths[idx],
                        'attr_labels': torch.zeros_like(self.attr_labels[idx])
                    }
                # 随机选择另一个样本
                idx = random.randint(0, len(self) - 1)

def create_data_loaders(config):
    """创建训练和验证数据加载器"""
    logger.info("开始创建数据加载器...")
    
    try:
        # 创建数据集实例
        logger.info("创建训练集...")
        train_dataset = DeepFashionDataset(
            img_list_file=config['data']['train_img_list_file'],
            attr_file=config['data']['train_attr_file'],
            cate_file=config['data']['train_cate_file'],
            bbox_file=config['data']['train_bbox_file'],
            seg_file=config['data']['train_seg_file'],
            image_dir=config['data']['root_dir'],
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ]),
            max_retry=config['data']['max_retry']
        )
        
        logger.info("创建验证集...")
        val_dataset = DeepFashionDataset(
            img_list_file=config['data']['val_img_list_file'],
            attr_file=config['data']['val_attr_file'],
            cate_file=config['data']['val_cate_file'],
            bbox_file=config['data']['val_bbox_file'],
            seg_file=config['data']['val_seg_file'],
            image_dir=config['data']['root_dir'],
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ]),
            max_retry=config['data']['max_retry']
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("数据集为空")
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        logger.info("数据加载器创建成功")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        raise 
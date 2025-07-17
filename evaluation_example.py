import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from evaluation import FashionMultiLabelEvaluator
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_model(num_classes=26):
    """创建测试模型"""
    class TestModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(64, num_classes)
            
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return {'attr_logits': x}
    
    return TestModel(num_classes)

def create_test_dataset(num_samples=100, num_attrs=26):
    """创建测试数据集"""
    class TestDataset:
        def __init__(self, num_samples, num_attrs):
            self.num_samples = num_samples
            self.num_attrs = num_attrs
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            image = torch.randn(3, 224, 224)
            labels = torch.zeros(self.num_attrs)
            num_positives = torch.randint(1, 6, (1,)).item()
            positive_indices = torch.randperm(self.num_attrs)[:num_positives]
            labels[positive_indices] = 1
            
            return {
                'image': image,
                'attr_labels': labels,
                'img_name': f'sample_{idx}.jpg'
            }
    
    return TestDataset(num_samples, num_attrs)

def create_test_dataloader(dataset, batch_size=8):
    """创建测试数据加载器"""
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        labels = torch.stack([item['attr_labels'] for item in batch])
        names = [item['img_name'] for item in batch]
        
        return {
            'image': images,
            'attr_labels': labels,
            'img_name': names
        }
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def test_optimized_evaluation():
    """测试优化后的评估系统"""
    logger.info("开始测试优化后的评估系统...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_attrs = 26
    attr_names = [
        'floral', 'graphic', 'striped', 'embroidered', 'pleated', 'solid', 'lattice',
        'long_sleeve', 'short_sleeve', 'sleeveless', 'maxi_length', 'mini_length',
        'no_dress', 'crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline',
        'denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit', 'tight', 'loose', 'conventional'
    ]
    
    # 创建模型和数据
    model = create_test_model(num_attrs)
    dataset = create_test_dataset(100, num_attrs)
    dataloader = create_test_dataloader(dataset, batch_size=8)
    
    # 创建评估器
    evaluator = FashionMultiLabelEvaluator(model, dataloader, device, attr_names)
    
    # 执行评估
    metrics = evaluator.evaluate(save_dir="optimized_evaluation_results")
    
    # 生成报告
    report_path = evaluator.generate_evaluation_report(metrics, "optimized_evaluation_results")
    
    logger.info(f"评估完成，报告保存在: {report_path}")
    
    # 打印关键指标
    logger.info("优化后的评估指标:")
    logger.info(f"准确率: {metrics['basic_metrics']['accuracy']:.4f}")
    logger.info(f"F1分数: {metrics['basic_metrics']['f1']:.4f}")
    logger.info(f"Hamming Loss: {metrics['basic_metrics']['hamming_loss']:.4f}")
    logger.info(f"子集准确率: {metrics['basic_metrics']['subset_accuracy']:.4f}")
    logger.info(f"平均ROC AUC: {metrics['advanced_metrics']['mean_roc_auc']:.4f}")
    logger.info(f"平均AP: {metrics['advanced_metrics']['mean_ap']:.4f}")
    
    # 保存结果
    with open('optimized_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics

if __name__ == "__main__":
    test_optimized_evaluation() 
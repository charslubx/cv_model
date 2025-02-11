# 新增train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

from base_model import FullModel


# ---------------------- 数据加载模块 ----------------------
class MultiLabelDataset(Dataset):
    """多标签图像数据集"""
    def __init__(self,
                 csv_path: str,
                 image_dir: str,
                 transform=None,
                 label_columns: list = None):
        """
        csv_path: 包含图像路径和标签的CSV文件
        image_dir: 图像存储根目录
        transform: 数据增强
        label_columns: 指定哪些列是标签
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.label_columns = label_columns or self.df.columns[1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row['image_path']}"
        image = Image.open(img_path).convert('RGB')

        # 转换为多热编码标签
        labels = torch.FloatTensor(row[self.label_columns].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, labels

# ---------------------- 训练模块 ----------------------
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 多标签分类使用BCEWithLogitsLoss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                # 计算准确率（阈值0.5）
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
                total += images.size(0)

        return total_loss / len(self.val_loader.dataset), correct / total

    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ---------------------- 数据增强配置 ----------------------
def get_transforms(train: bool = True):
    base_transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            *base_transform
        ])
    return transforms.Compose(base_transform)

# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 1. 准备数据
    train_dataset = MultiLabelDataset(
        csv_path="data/train_labels.csv",
        image_dir="data/images",
        transform=get_transforms(train=True),
        label_columns=[f"class_{i}" for i in range(20)]
    )

    val_dataset = MultiLabelDataset(
        csv_path="data/val_labels.csv",
        image_dir="data/images",
        transform=get_transforms(train=False),
        label_columns=[f"class_{i}" for i in range(20)]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 2. 初始化模型
    model = FullModel(num_classes=20)

    # 3. 开始训练
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=20)

    # 4. 保存模型
    torch.save(model.state_dict(), "multilabel_model.pth")
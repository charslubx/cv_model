"""
客户端调用示例
演示如何调用服装属性识别服务
"""
import requests
from pathlib import Path


class FashionAttrClient:
    """服装属性识别客户端"""
    
    def __init__(self, base_url="http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: 服务地址
        """
        self.base_url = base_url
    
    def health_check(self):
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_attributes(self):
        """获取属性列表"""
        response = requests.get(f"{self.base_url}/attributes")
        return response.json()
    
    def predict(self, image_path, threshold=0.5):
        """
        单张图片多标签分类
        
        Args:
            image_path: 图片路径
            threshold: 分类阈值（0-1之间），默认0.5
            
        Returns:
            分类结果
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'threshold': threshold}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"分类失败: {response.text}")
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        批量多标签分类
        
        Args:
            image_paths: 图片路径列表
            threshold: 分类阈值（0-1之间），默认0.5
            
        Returns:
            分类结果列表
        """
        files = [('files', open(p, 'rb')) for p in image_paths]
        params = {'threshold': threshold}
        response = requests.post(
            f"{self.base_url}/predict_batch",
            files=files,
            params=params
        )
        
        # 关闭文件
        for _, f in files:
            f.close()
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"批量分类失败: {response.text}")


def main():
    """示例使用"""
    # 创建客户端
    client = FashionAttrClient()
    
    # 1. 健康检查
    print("健康检查:")
    health = client.health_check()
    print(f"  状态: {health['status']}")
    print(f"  模型已加载: {health['model_loaded']}")
    
    # 2. 获取属性列表
    print("\n获取属性列表:")
    attributes = client.get_attributes()
    print(f"  属性数量: {attributes['num_classes']}")
    print(f"  前5个属性: {attributes['attributes'][:5]}")
    
    # 3. 单张图片分类
    image_path = "../data/test_image.jpg"  # 请替换为实际路径
    if Path(image_path).exists():
        print(f"\n分类图片: {image_path}")
        
        # 使用默认阈值0.5
        result = client.predict(image_path)
        print(f"  文件名: {result['filename']}")
        stats = result['predictions']['statistics']
        print(f"  统计: 预测为正 {stats['positive_count']}/{stats['total_attributes']} 个属性")
        print(f"  预测为正的属性:")
        for attr in result['predictions']['positive_attributes'][:5]:
            print(f"    - {attr['attribute']}: {attr['confidence']:.2%}")
        
        # 尝试不同的阈值
        print(f"\n  使用阈值0.7:")
        result_high = client.predict(image_path, threshold=0.7)
        stats_high = result_high['predictions']['statistics']
        print(f"    预测为正 {stats_high['positive_count']}/{stats_high['total_attributes']} 个属性")
    else:
        print(f"\n图片不存在: {image_path}")
    
    # 4. 批量分类
    image_paths = [
        "../data/test_image1.jpg",  # 请替换为实际路径
        "../data/test_image2.jpg",
    ]
    valid_paths = [p for p in image_paths if Path(p).exists()]
    if valid_paths:
        print(f"\n批量分类 {len(valid_paths)} 张图片:")
        results = client.predict_batch(valid_paths, threshold=0.5)
        for i, result in enumerate(results['results']):
            if result['success']:
                print(f"\n  图片 {i+1}: {result['filename']}")
                stats = result['predictions']['statistics']
                print(f"    预测为正: {stats['positive_count']} 个属性")
                print(f"    Top-3属性:")
                for attr in result['predictions']['positive_attributes'][:3]:
                    print(f"      - {attr['attribute']}: {attr['confidence']:.2%}")
            else:
                print(f"\n  图片 {i+1}: {result['filename']} - 失败")


if __name__ == "__main__":
    main()

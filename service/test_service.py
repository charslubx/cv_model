"""
服务测试脚本
"""
import sys
import time
import requests
from pathlib import Path


def test_health():
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_attributes():
    """测试获取属性列表接口"""
    print("\n" + "=" * 50)
    print("测试获取属性列表接口")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:8000/attributes")
        print(f"状态码: {response.status_code}")
        data = response.json()
        print(f"属性数量: {data['num_classes']}")
        print(f"属性列表: {data['attributes'][:5]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_predict(image_path):
    """测试单张图片预测接口"""
    print("\n" + "=" * 50)
    print("测试单张图片预测接口")
    print("=" * 50)
    
    if not Path(image_path).exists():
        print(f"图片文件不存在: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post("http://localhost:8000/predict", files=files)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"文件名: {data['filename']}")
            print(f"Top-5属性:")
            for attr in data['predictions']['top_k_attributes']:
                print(f"  - {attr['attribute']}: {attr['confidence']:.4f}")
        else:
            print(f"错误: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_predict_batch(image_paths):
    """测试批量图片预测接口"""
    print("\n" + "=" * 50)
    print("测试批量图片预测接口")
    print("=" * 50)
    
    # 检查文件是否存在
    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        print("没有有效的图片文件")
        return False
    
    try:
        files = [('files', open(p, 'rb')) for p in valid_paths]
        response = requests.post("http://localhost:8000/predict_batch", files=files)
        
        # 关闭文件
        for _, f in files:
            f.close()
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"总数: {data['total']}")
            for result in data['results']:
                if result['success']:
                    print(f"\n文件: {result['filename']}")
                    print(f"Top-3属性:")
                    for attr in result['predictions']['top_k_attributes'][:3]:
                        print(f"  - {attr['attribute']}: {attr['confidence']:.4f}")
                else:
                    print(f"\n文件: {result['filename']} - 失败: {result['error']}")
        else:
            print(f"错误: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("服装属性识别服务测试")
    print("=" * 50)
    print("请确保服务已启动: python app.py")
    print("=" * 50)
    
    # 等待用户确认
    input("按Enter键开始测试...")
    
    # 等待服务启动
    print("\n等待服务启动...")
    for i in range(5):
        try:
            requests.get("http://localhost:8000/")
            print("服务已就绪!")
            break
        except:
            if i == 4:
                print("无法连接到服务，请确认服务是否正在运行")
                return
            time.sleep(1)
    
    # 运行测试
    results = []
    
    # 1. 健康检查
    results.append(("健康检查", test_health()))
    
    # 2. 获取属性列表
    results.append(("属性列表", test_attributes()))
    
    # 3. 单张图片预测 (需要提供测试图片路径)
    # 这里使用一个示例路径，实际使用时需要替换
    test_image = "../data/test_image.jpg"  # 请替换为实际图片路径
    results.append(("单张预测", test_predict(test_image)))
    
    # 4. 批量预测
    test_images = [
        "../data/test_image1.jpg",  # 请替换为实际图片路径
        "../data/test_image2.jpg",
    ]
    results.append(("批量预测", test_predict_batch(test_images)))
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
    
    # 总结
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\n总计: {passed}/{total} 测试通过")


if __name__ == "__main__":
    main()

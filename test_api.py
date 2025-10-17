"""
API测试脚本
使用示例：python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"


def test_health():
    """测试健康检查"""
    print("测试健康检查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_load_model():
    """测试加载模型"""
    print("测试加载模型...")
    response = requests.post(f"{BASE_URL}/model/load")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_predict():
    """测试单个推理"""
    print("测试单个样本推理...")
    data = {
        "data": "test_input",
        "batch_size": 1
    }
    response = requests.post(f"{BASE_URL}/model/predict", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_batch_predict():
    """测试批量推理"""
    print("测试批量推理...")
    data = {
        "data_list": ["input1", "input2", "input3"],
        "batch_size": 32
    }
    response = requests.post(f"{BASE_URL}/model/batch_predict", json=data)
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


def test_unload_model():
    """测试卸载模型"""
    print("测试卸载模型...")
    response = requests.post(f"{BASE_URL}/model/unload")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("FastAPI模型服务测试")
    print("=" * 50)
    print()
    
    try:
        # 测试健康检查
        test_health()
        
        # 测试加载模型
        test_load_model()
        
        # 测试单个推理
        test_predict()
        
        # 测试批量推理
        test_batch_predict()
        
        # 测试卸载模型
        test_unload_model()
        
        # 再次检查健康状态
        test_health()
        
        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("错误：无法连接到服务器")
        print("请确保服务已启动: python run.py")
    except Exception as e:
        print(f"测试出错: {str(e)}")

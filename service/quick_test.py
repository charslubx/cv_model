"""
快速测试脚本 - 验证服务组件
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("服装属性识别服务 - 快速测试")
print("=" * 60)

# 1. 测试配置导入
print("\n[1/5] 测试配置模块...")
try:
    from service.config import Settings
    settings = Settings()
    print(f"  ✅ 配置加载成功")
    print(f"     - HOST: {settings.HOST}")
    print(f"     - PORT: {settings.PORT}")
    print(f"     - DEVICE: {settings.DEVICE}")
    print(f"     - NUM_CLASSES: {settings.NUM_CLASSES}")
except Exception as e:
    print(f"  ❌ 配置加载失败: {str(e)}")
    sys.exit(1)

# 2. 测试模型导入
print("\n[2/5] 测试模型导入...")
try:
    from base_model import FullModel
    print(f"  ✅ FullModel导入成功")
except Exception as e:
    print(f"  ❌ 模型导入失败: {str(e)}")
    sys.exit(1)

# 3. 测试推理模块
print("\n[3/5] 测试推理模块...")
try:
    from service.model_loader import ModelInference
    print(f"  ✅ ModelInference导入成功")
except Exception as e:
    print(f"  ❌ 推理模块导入失败: {str(e)}")
    sys.exit(1)

# 4. 测试FastAPI应用
print("\n[4/5] 测试FastAPI应用...")
try:
    from service.app import app
    print(f"  ✅ FastAPI应用导入成功")
    print(f"     - Title: {app.title}")
    print(f"     - Version: {app.version}")
except Exception as e:
    print(f"  ❌ FastAPI应用导入失败: {str(e)}")
    sys.exit(1)

# 5. 测试模型初始化（不加载权重）
print("\n[5/5] 测试模型初始化...")
try:
    import torch
    model_inference = ModelInference(
        model_path=None,  # 不加载权重
        device="cpu",     # 使用CPU
        num_classes=26
    )
    print(f"  ✅ 模型初始化成功")
    print(f"     - 设备: {model_inference.device}")
    print(f"     - 类别数: {model_inference.num_classes}")
    print(f"     - 属性数量: {len(model_inference.get_attribute_names())}")
    
    # 测试预测功能（使用随机图片）
    from PIL import Image
    import numpy as np
    test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    result = model_inference.predict(test_img)
    print(f"  ✅ 推理测试成功")
    print(f"     - 返回属性数: {len(result['attributes'])}")
    print(f"     - Top-5属性: {[attr['attribute'] for attr in result['top_k_attributes']]}")
    
except Exception as e:
    print(f"  ❌ 模型初始化失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！服务组件正常工作")
print("=" * 60)
print("\n下一步:")
print("1. 训练模型并保存权重到 checkpoints/best_model.pth")
print("2. 启动服务: cd service && python3 app.py")
print("3. 或使用脚本启动: bash service/start_service.sh")
print("=" * 60)

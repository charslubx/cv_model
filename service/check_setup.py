"""
检查服务设置
不需要依赖项，仅检查文件结构
"""
from pathlib import Path

print("=" * 60)
print("服装属性识别服务 - 设置检查")
print("=" * 60)

# 定义需要检查的文件
required_files = [
    "service/__init__.py",
    "service/app.py",
    "service/config.py",
    "service/model_loader.py",
    "service/requirements_service.txt",
    "service/README.md",
    "service/start_service.sh",
    "service/test_service.py",
    "service/client_example.py",
    "service/Dockerfile",
    "service/.env.example",
]

# 检查文件是否存在
print("\n检查文件:")
all_exist = True
for file in required_files:
    path = Path(file)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {file}")
    if not exists:
        all_exist = False

# 检查start_service.sh是否可执行
print("\n检查权限:")
start_script = Path("service/start_service.sh")
if start_script.exists():
    import os
    is_executable = os.access(start_script, os.X_OK)
    status = "✅" if is_executable else "⚠️ "
    print(f"  {status} start_service.sh 可执行权限")

# 总结
print("\n" + "=" * 60)
if all_exist:
    print("✅ 所有必要文件已创建！")
    print("\n服务已准备就绪，包含以下组件:")
    print("  - FastAPI应用主文件")
    print("  - 模型加载和推理模块")
    print("  - 配置管理")
    print("  - 测试脚本")
    print("  - 客户端示例")
    print("  - Docker支持")
    print("  - 启动脚本")
    print("  - 完整文档")
else:
    print("❌ 部分文件缺失")

print("=" * 60)

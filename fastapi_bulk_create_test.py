"""
FastAPI 批量创建功能测试脚本

这个脚本用于测试各种批量创建方法的性能和正确性
"""

import requests
import time
import json
from typing import List, Dict
import random


BASE_URL = "http://localhost:8000"


def generate_test_products(count: int) -> List[Dict]:
    """生成测试用的产品数据"""
    categories = ["Electronics", "Clothing", "Books", "Food", "Toys"]
    products = []
    
    for i in range(count):
        products.append({
            "name": f"Product {i+1}",
            "category": random.choice(categories),
            "price": round(random.uniform(10, 1000), 2),
            "stock": random.randint(0, 1000),
            "description": f"This is product number {i+1}"
        })
    
    return products


def generate_test_users(count: int) -> List[Dict]:
    """生成测试用的用户数据"""
    users = []
    
    for i in range(count):
        users.append({
            "username": f"user{i+1}",
            "email": f"user{i+1}@example.com",
            "full_name": f"User Number {i+1}"
        })
    
    return users


def clear_products():
    """清空产品表"""
    try:
        response = requests.delete(f"{BASE_URL}/products/clear")
        print("✓ 产品表已清空")
    except Exception as e:
        print(f"✗ 清空产品表失败: {e}")


def test_bulk_create_method(
    method_name: str,
    endpoint: str,
    products: List[Dict],
    description: str
):
    """测试某个批量创建方法"""
    print(f"\n{'='*70}")
    print(f"测试方法: {method_name}")
    print(f"说明: {description}")
    print(f"数据量: {len(products)} 条")
    print(f"{'='*70}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=products,
            headers={"Content-Type": "application/json"}
        )
        client_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 成功")
            print(f"  - 创建数量: {result['created_count']}")
            print(f"  - 失败数量: {result['failed_count']}")
            print(f"  - 服务器执行时间: {result['execution_time']}秒")
            print(f"  - 客户端总耗时: {round(client_time, 4)}秒")
            print(f"  - 吞吐量: {round(result['created_count']/result['execution_time'], 2)} 条/秒")
            return result
        else:
            print(f"✗ 失败: {response.status_code}")
            print(f"  错误信息: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ 异常: {str(e)}")
        return None


def test_all_methods():
    """测试所有批量创建方法"""
    print("\n" + "="*70)
    print("FastAPI 批量创建性能测试")
    print("="*70)
    
    # 测试不同数据量
    test_sizes = [10, 100, 1000]
    
    methods = [
        {
            "name": "Method 1 - bulk_save_objects",
            "endpoint": "/products/bulk-create/method1",
            "description": "使用 SQLAlchemy bulk_save_objects"
        },
        {
            "name": "Method 2 - bulk_insert_mappings",
            "endpoint": "/products/bulk-create/method2",
            "description": "使用 SQLAlchemy bulk_insert_mappings (推荐)"
        },
        {
            "name": "Method 3 - 批量添加",
            "endpoint": "/products/bulk-create/method3",
            "description": "逐个添加到会话，批量提交"
        },
        {
            "name": "Method 4 - 原生SQL",
            "endpoint": "/products/bulk-create/method4",
            "description": "使用原生 SQL 批量插入"
        },
        {
            "name": "Method 5 - 分批插入",
            "endpoint": "/products/bulk-create/method5",
            "description": "分批插入，适合大数据量"
        },
    ]
    
    results = {}
    
    for size in test_sizes:
        print(f"\n\n{'#'*70}")
        print(f"# 测试数据量: {size} 条")
        print(f"{'#'*70}")
        
        results[size] = {}
        
        for method in methods:
            # 清空表
            clear_products()
            time.sleep(0.5)  # 等待清空完成
            
            # 生成测试数据
            products = generate_test_products(size)
            
            # 测试方法
            result = test_bulk_create_method(
                method["name"],
                method["endpoint"],
                products,
                method["description"]
            )
            
            if result:
                results[size][method["name"]] = {
                    "execution_time": result["execution_time"],
                    "throughput": round(
                        result['created_count']/result['execution_time'], 2
                    )
                }
            
            time.sleep(1)  # 等待一秒再测试下一个方法
    
    # 打印性能对比表
    print("\n\n" + "="*70)
    print("性能对比汇总")
    print("="*70)
    print(f"{'数据量':<12} {'方法':<35} {'耗时(秒)':<12} {'吞吐量(条/秒)'}")
    print("-"*70)
    
    for size in test_sizes:
        for method_name, data in results.get(size, {}).items():
            print(
                f"{size:<12} "
                f"{method_name:<35} "
                f"{data['execution_time']:<12} "
                f"{data['throughput']}"
            )
        print("-"*70)


def test_users_bulk_create():
    """测试用户批量创建"""
    print("\n" + "="*70)
    print("测试用户批量创建")
    print("="*70)
    
    users = generate_test_users(100)
    
    try:
        response = requests.post(
            f"{BASE_URL}/users/bulk-create",
            json=users,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 成功创建 {result['created_count']} 个用户")
            print(f"  执行时间: {result['execution_time']}秒")
        else:
            print(f"✗ 失败: {response.text}")
            
    except Exception as e:
        print(f"✗ 异常: {str(e)}")


def verify_data():
    """验证插入的数据"""
    print("\n" + "="*70)
    print("验证数据")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/products?limit=5")
        if response.status_code == 200:
            products = response.json()
            print(f"✓ 查询到 {len(products)} 条产品数据")
            print("\n前5条数据示例:")
            for i, product in enumerate(products[:5], 1):
                print(f"\n{i}. {product['name']}")
                print(f"   类别: {product['category']}")
                print(f"   价格: ${product['price']}")
                print(f"   库存: {product['stock']}")
    except Exception as e:
        print(f"✗ 查询失败: {str(e)}")


def main():
    """主函数"""
    print("""
    =============================================================
    FastAPI 批量创建功能测试
    =============================================================
    
    请确保 FastAPI 服务已经启动在 http://localhost:8000
    
    测试内容：
    1. 测试所有批量创建方法
    2. 对比不同数据量下的性能
    3. 验证数据正确性
    
    =============================================================
    """)
    
    input("按 Enter 键开始测试...")
    
    # 测试所有方法
    test_all_methods()
    
    # 测试用户批量创建
    test_users_bulk_create()
    
    # 验证数据
    verify_data()
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()

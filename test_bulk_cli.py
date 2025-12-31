#!/usr/bin/env python3
"""
FastAPI 批量创建命令行测试工具

这个脚本提供了一个简单的命令行界面来测试批量创建功能
"""

import argparse
import requests
import json
import time
import random
from typing import List, Dict


def generate_products(count: int) -> List[Dict]:
    """生成测试产品数据"""
    categories = ["电子产品", "服装", "图书", "食品", "玩具", "家具", "运动", "美妆"]
    products = []
    
    for i in range(count):
        products.append({
            "name": f"测试产品 {i+1}",
            "category": random.choice(categories),
            "price": round(random.uniform(10, 2000), 2),
            "stock": random.randint(0, 1000),
            "description": f"这是第 {i+1} 个测试产品"
        })
    
    return products


def bulk_create(base_url: str, products: List[Dict], method: str = "method2") -> Dict:
    """
    执行批量创建
    
    Args:
        base_url: API 基础 URL
        products: 产品列表
        method: 批量创建方法 (method1-method6)
    
    Returns:
        响应数据
    """
    url = f"{base_url}/products/bulk-create/{method}"
    
    print(f"\n📤 发送请求到: {url}")
    print(f"📦 数据量: {len(products)} 条")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            url,
            json=products,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5分钟超时
        )
        
        client_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            result["client_time"] = round(client_time, 4)
            return result
        else:
            return {
                "error": True,
                "status_code": response.status_code,
                "message": response.text
            }
    
    except requests.exceptions.Timeout:
        return {"error": True, "message": "请求超时"}
    except requests.exceptions.ConnectionError:
        return {"error": True, "message": "无法连接到服务器"}
    except Exception as e:
        return {"error": True, "message": str(e)}


def get_products(base_url: str, limit: int = 10) -> List[Dict]:
    """获取产品列表"""
    try:
        response = requests.get(f"{base_url}/products?limit={limit}")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def clear_products(base_url: str) -> bool:
    """清空产品表"""
    try:
        response = requests.delete(f"{base_url}/products/clear")
        return response.status_code == 200
    except Exception:
        return False


def print_result(result: Dict):
    """打印结果"""
    print("\n" + "="*70)
    print("📊 批量创建结果")
    print("="*70)
    
    if result.get("error"):
        print(f"❌ 失败: {result.get('message')}")
        if "status_code" in result:
            print(f"   状态码: {result['status_code']}")
    else:
        print(f"✅ 成功: {result.get('message', '批量创建完成')}")
        print(f"   创建数量: {result.get('created_count', 0)}")
        print(f"   失败数量: {result.get('failed_count', 0)}")
        print(f"   服务器耗时: {result.get('execution_time', 0)} 秒")
        print(f"   客户端耗时: {result.get('client_time', 0)} 秒")
        
        if result.get('execution_time', 0) > 0:
            throughput = result.get('created_count', 0) / result.get('execution_time', 1)
            print(f"   吞吐量: {round(throughput, 2)} 条/秒")
    
    print("="*70)


def print_products(products: List[Dict]):
    """打印产品列表"""
    if not products:
        print("\n📭 没有产品数据")
        return
    
    print(f"\n📦 产品列表（共 {len(products)} 条）")
    print("-"*70)
    
    for i, product in enumerate(products, 1):
        print(f"{i}. {product['name']}")
        print(f"   类别: {product['category']}")
        print(f"   价格: ¥{product['price']}")
        print(f"   库存: {product.get('stock', 'N/A')}")
        print()


def performance_test(base_url: str, sizes: List[int], methods: List[str]):
    """性能测试"""
    print("\n" + "="*70)
    print("🚀 批量创建性能测试")
    print("="*70)
    
    results = {}
    
    for size in sizes:
        print(f"\n\n{'#'*70}")
        print(f"# 测试数据量: {size} 条")
        print(f"{'#'*70}")
        
        results[size] = {}
        
        for method in methods:
            print(f"\n⏳ 测试 {method}...")
            
            # 清空表
            if clear_products(base_url):
                print("   ✓ 表已清空")
            
            time.sleep(0.5)
            
            # 生成数据
            products = generate_products(size)
            
            # 执行批量创建
            result = bulk_create(base_url, products, method)
            
            if not result.get("error"):
                results[size][method] = {
                    "execution_time": result.get("execution_time", 0),
                    "throughput": round(
                        result.get('created_count', 0) / result.get('execution_time', 1),
                        2
                    )
                }
                print(f"   ✓ 耗时: {result.get('execution_time')}秒")
                print(f"   ✓ 吞吐量: {results[size][method]['throughput']} 条/秒")
            else:
                print(f"   ✗ 失败: {result.get('message')}")
            
            time.sleep(1)
    
    # 打印性能对比表
    print("\n\n" + "="*70)
    print("📊 性能对比汇总")
    print("="*70)
    print(f"{'数据量':<12} {'方法':<15} {'耗时(秒)':<12} {'吞吐量(条/秒)'}")
    print("-"*70)
    
    for size in sizes:
        for method, data in results.get(size, {}).items():
            print(
                f"{size:<12} "
                f"{method:<15} "
                f"{data['execution_time']:<12} "
                f"{data['throughput']}"
            )
        print("-"*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FastAPI 批量创建命令行测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 批量创建 100 条数据
  python test_bulk_cli.py create -n 100
  
  # 使用特定方法批量创建
  python test_bulk_cli.py create -n 1000 -m method2
  
  # 查看产品列表
  python test_bulk_cli.py list -l 20
  
  # 清空产品表
  python test_bulk_cli.py clear
  
  # 性能测试
  python test_bulk_cli.py perf -s 100 1000 5000
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API 基础 URL (默认: http://localhost:8000)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # create 命令
    create_parser = subparsers.add_parser("create", help="批量创建产品")
    create_parser.add_argument(
        "-n", "--count",
        type=int,
        default=100,
        help="创建的产品数量 (默认: 100)"
    )
    create_parser.add_argument(
        "-m", "--method",
        default="method2",
        choices=["method1", "method2", "method3", "method4", "method5", "method6"],
        help="批量创建方法 (默认: method2)"
    )
    
    # list 命令
    list_parser = subparsers.add_parser("list", help="查看产品列表")
    list_parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="显示的产品数量 (默认: 10)"
    )
    
    # clear 命令
    subparsers.add_parser("clear", help="清空产品表")
    
    # perf 命令
    perf_parser = subparsers.add_parser("perf", help="性能测试")
    perf_parser.add_argument(
        "-s", "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="测试的数据量 (默认: 100 1000)"
    )
    perf_parser.add_argument(
        "-m", "--methods",
        nargs="+",
        default=["method1", "method2", "method5"],
        help="测试的方法 (默认: method1 method2 method5)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行命令
    if args.command == "create":
        print(f"\n🚀 开始批量创建 {args.count} 条产品...")
        products = generate_products(args.count)
        result = bulk_create(args.url, products, args.method)
        print_result(result)
        
    elif args.command == "list":
        print(f"\n📋 获取产品列表...")
        products = get_products(args.url, args.limit)
        print_products(products)
        
    elif args.command == "clear":
        print(f"\n🗑️  清空产品表...")
        if clear_products(args.url):
            print("✅ 产品表已清空")
        else:
            print("❌ 清空失败")
        
    elif args.command == "perf":
        performance_test(args.url, args.sizes, args.methods)


if __name__ == "__main__":
    main()

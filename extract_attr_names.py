"""
提取并显示DeepFashion数据集的属性名称
用于确认模型使用的属性列表

使用方法:
    python extract_attr_names.py
    python extract_attr_names.py --dataset-root /path/to/deepfashion
    python extract_attr_names.py --output attr_names.txt
"""

import argparse
import os
import sys
import json


def extract_attr_names(dataset_root="/home/cv_model/deepfashion"):
    """
    从DeepFashion数据集提取属性名称
    
    Args:
        dataset_root: DeepFashion数据集根目录
        
    Returns:
        属性定义列表
    """
    attr_file = os.path.join(
        dataset_root,
        "Category and Attribute Prediction Benchmark",
        "Anno_fine",
        "list_attr_cloth.txt"
    )
    
    if not os.path.exists(attr_file):
        print(f"错误: 属性定义文件不存在: {attr_file}")
        print(f"请检查数据集路径是否正确")
        return None
    
    try:
        with open(attr_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 第一行是属性数量
        num_attrs = int(lines[0].strip())
        print(f"属性数量: {num_attrs}")
        
        # 第二行是表头
        header = lines[1].strip()
        print(f"表头: {header}")
        
        # 从第三行开始是属性定义
        attrs = []
        for i, line in enumerate(lines[2:2 + num_attrs], 1):
            parts = line.strip().split()
            if len(parts) >= 2:
                attr_name = parts[0]
                attr_type = int(parts[1])
                
                # 属性类型映射
                type_map = {
                    1: "纹理(texture)",
                    2: "面料(fabric)",
                    3: "形状(shape)",
                    4: "部件(part)",
                    5: "风格(style)",
                    6: "合身度(fit)"
                }
                
                attrs.append({
                    'index': i - 1,  # 从0开始的索引
                    'name': attr_name,
                    'type': attr_type,
                    'type_name': type_map.get(attr_type, "未知")
                })
        
        print(f"成功提取 {len(attrs)} 个属性定义")
        return attrs
        
    except Exception as e:
        print(f"读取属性文件失败: {str(e)}")
        return None


def print_attrs(attrs, show_type=True):
    """打印属性列表"""
    if not attrs:
        return
    
    print("\n" + "=" * 80)
    print("DeepFashion 属性列表")
    print("=" * 80)
    
    if show_type:
        # 按类型分组显示
        from collections import defaultdict
        type_groups = defaultdict(list)
        for attr in attrs:
            type_groups[attr['type_name']].append(attr)
        
        for type_name, group in sorted(type_groups.items()):
            print(f"\n【{type_name}】 ({len(group)}个属性)")
            print("-" * 80)
            for attr in group:
                print(f"  [{attr['index']:2d}] {attr['name']}")
    else:
        # 简单列表显示
        for attr in attrs:
            print(f"[{attr['index']:2d}] {attr['name']}")
    
    print("\n" + "=" * 80)


def save_to_file(attrs, output_path, format='txt'):
    """保存属性列表到文件"""
    if not attrs:
        return
    
    try:
        if format == 'txt':
            # 保存为简单文本列表
            with open(output_path, 'w', encoding='utf-8') as f:
                for attr in attrs:
                    f.write(f"{attr['name']}\n")
            print(f"✓ 属性名称已保存到: {output_path} (文本格式)")
            
        elif format == 'json':
            # 保存为JSON格式（包含详细信息）
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(attrs, f, indent=2, ensure_ascii=False)
            print(f"✓ 属性详情已保存到: {output_path} (JSON格式)")
            
        elif format == 'python':
            # 保存为Python列表
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# DeepFashion 属性名称列表\n")
                f.write("# 自动生成，请勿手动修改\n\n")
                f.write("ATTR_NAMES = [\n")
                for attr in attrs:
                    f.write(f"    '{attr['name']}',  # [{attr['index']}] {attr['type_name']}\n")
                f.write("]\n")
            print(f"✓ 属性列表已保存到: {output_path} (Python格式)")
            
    except Exception as e:
        print(f"保存文件失败: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='提取DeepFashion数据集的属性名称')
    
    parser.add_argument('--dataset-root', type=str, 
                        default='/home/cv_model/deepfashion',
                        help='DeepFashion数据集根目录')
    parser.add_argument('--output', type=str,
                        help='输出文件路径（可选）')
    parser.add_argument('--format', type=str, 
                        choices=['txt', 'json', 'python'],
                        default='txt',
                        help='输出格式: txt(简单列表), json(详细信息), python(Python代码)')
    parser.add_argument('--no-group', action='store_true',
                        help='不按类型分组显示')
    
    args = parser.parse_args()
    
    # 提取属性
    print(f"正在从数据集提取属性名称...")
    print(f"数据集路径: {args.dataset_root}")
    print("-" * 80)
    
    attrs = extract_attr_names(args.dataset_root)
    
    if not attrs:
        print("\n❌ 提取失败")
        sys.exit(1)
    
    # 显示属性
    print_attrs(attrs, show_type=not args.no_group)
    
    # 保存到文件（如果指定）
    if args.output:
        save_to_file(attrs, args.output, args.format)
    
    # 显示使用提示
    print("\n" + "=" * 80)
    print("使用方法:")
    print("=" * 80)
    print("\n1. 在Python代码中使用:")
    print("   from inference import get_attr_names_from_dataset")
    print(f"   attr_names = get_attr_names_from_dataset('{args.dataset_root}')")
    
    print("\n2. 使用推理工具时:")
    attr_file = os.path.join(
        args.dataset_root,
        "Category and Attribute Prediction Benchmark",
        "Anno_fine",
        "list_attr_cloth.txt"
    )
    print(f"   python quick_inference.py --model model.pth --image test.jpg \\")
    print(f"       --attr-file {attr_file}")
    
    print("\n3. 手动创建属性列表:")
    print("   attr_names = [")
    for attr in attrs[:3]:
        print(f"       '{attr['name']}',")
    print("       ...")
    print("   ]")
    print("=" * 80)


if __name__ == "__main__":
    main()

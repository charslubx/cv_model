#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动混合数据集训练脚本

简化的启动脚本，用于快速开始混合数据集训练
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment():
    """检查训练环境"""
    logger.info("检查训练环境...")
    
    # 检查必要的文件
    required_files = [
        'base_model.py',
        'training.py',
        'integrated_training.py'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"缺少必要文件: {missing_files}")
        return False
    
    # 检查数据集路径
    data_paths = [
        "/home/cv_model/fabric/train",
        "/home/cv_model/fiber/train"
    ]
    
    available_datasets = []
    for path in data_paths:
        if os.path.exists(path):
            available_datasets.append(path)
            logger.info(f"✓ 找到数据集: {path}")
        else:
            logger.warning(f"✗ 数据集不存在: {path}")
    
    if not available_datasets:
        logger.error("没有找到任何可用的数据集!")
        return False
    
    logger.info(f"环境检查完成，找到 {len(available_datasets)} 个数据集")
    return True


def start_training():
    """启动训练"""
    try:
        # 检查环境
        if not check_environment():
            logger.error("环境检查失败，无法启动训练")
            return 1
        
        # 导入并运行集成训练
        logger.info("启动集成训练...")
        from integrated_training import main
        
        return main()
        
    except ImportError as e:
        logger.error(f"导入模块失败: {e}")
        logger.info("请确保所有必要的文件都存在")
        return 1
    except Exception as e:
        logger.error(f"训练启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    logger.info("🚀 开始混合数据集训练...")
    result = start_training()
    
    if result == 0:
        logger.info("🎉 训练成功完成!")
    else:
        logger.error("❌ 训练失败!")
    
    exit(result)

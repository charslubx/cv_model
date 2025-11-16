#!/usr/bin/env python
"""
论文实验快速启动脚本
提供交互式菜单，方便选择和运行不同类型的实验
"""

import sys
import os
import torch
from datetime import datetime


def print_banner():
    """打印欢迎界面"""
    print("\n" + "="*80)
    print(" " * 25 + "论文实验系统")
    print(" " * 15 + "服装属性识别 AdaGAT 方法完整实验框架")
    print("="*80 + "\n")


def check_environment():
    """检查运行环境"""
    print("正在检查运行环境...")
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA 不可用，将使用 CPU（速度会很慢）")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return False
    
    # 检查必要文件
    required_files = [
        'ablation_models.py',
        'lambda_experiment.py',
        'comprehensive_experiments.py',
        'dataset.py',
        'experiment_config.py'
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"\n✗ 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    print("✓ 所有必要文件存在")
    
    # 检查数据集配置
    try:
        from experiment_config import EXPERIMENT_CONFIG, validate_dataset_structure
        if validate_dataset_structure():
            print("✓ 数据集配置正确")
        else:
            print("✗ 数据集配置有误，请检查路径")
            return False
    except Exception as e:
        print(f"✗ 检查数据集时出错: {str(e)}")
        return False
    
    print("\n环境检查完成！\n")
    return True


def show_menu():
    """显示菜单"""
    print("\n" + "="*80)
    print("请选择要运行的实验：")
    print("="*80)
    print("\n[1] 快速测试 - 验证框架是否正常工作（推荐首次运行）")
    print("[2] λ 超参数实验 - 测试 λ = 0, 0.3, 0.5, 0.7, 1.0 的影响")
    print("[3] 模块级消融实验 - 验证各模块的有效性")
    print("[4] SOTA 对比实验 - 与现有方法对比")
    print("[5] 完整实验套件 - 运行所有实验（耗时较长）")
    print("[6] 自定义配置 - 自己设置实验参数")
    print("[0] 退出")
    print("\n" + "="*80)


def run_quick_test():
    """运行快速测试"""
    print("\n" + "="*80)
    print("运行快速测试...")
    print("="*80 + "\n")
    
    import test_experiment_framework
    return test_experiment_framework.main()


def run_lambda_experiment(epochs=30, lr=3e-4):
    """运行 λ 实验"""
    print("\n" + "="*80)
    print("λ 超参数实验选项")
    print("="*80)
    print("\n[1] 基于预训练模型评估（推荐，快速）")
    print("    - 加载已训练好的模型")
    print("    - 直接在验证集上测试不同λ值")
    print("    - 不需要重新训练")
    print("\n[2] 从头训练不同λ值的模型")
    print("    - 对每个λ值从头训练模型")
    print(f"    - 需要训练 {epochs} 个epochs")
    print("    - 耗时较长")
    print("\n" + "="*80)
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == '1':
        # 基于预训练模型的快速评估
        print("\n使用预训练模型进行 λ 评估...")
        
        from lambda_eval_experiment import LambdaEvaluationExperiment
        from dataset import create_data_loaders
        from experiment_config import EXPERIMENT_CONFIG
        import os
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
        
        # 检查预训练模型
        pretrained_model_path = 'smart_mixed_checkpoints/best_model.pth'
        if not os.path.exists(pretrained_model_path):
            print(f"\n✗ 找不到预训练模型: {pretrained_model_path}")
            print("请先训练模型或选择选项[2]从头训练")
            return 1
        
        experiment = LambdaEvaluationExperiment(
            pretrained_model_path=pretrained_model_path,
            val_loader=val_loader,
            device=device,
            num_classes=26
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'lambda_eval_results_{timestamp}'
        
        results = experiment.run_experiment(save_dir=save_dir)
        
        print(f"\n实验完成！结果保存在: {save_dir}")
        return 0
        
    elif choice == '2':
        # 从头训练
        print(f"\n从头训练不同λ值的模型（{epochs} epochs）...")
        
        from lambda_experiment import LambdaExperiment
        from dataset import create_data_loaders
        from experiment_config import EXPERIMENT_CONFIG
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
        
        lambda_exp = LambdaExperiment(train_loader, val_loader, device, num_classes=26)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'lambda_train_results_{timestamp}'
        
        results = lambda_exp.run_experiment(
            epochs=epochs,
            learning_rate=lr,
            save_dir=save_dir
        )
        
        print(f"\n实验完成！结果保存在: {save_dir}")
        return 0
    else:
        print("\n无效选项")
        return 1


def run_ablation_experiment(epochs=30, lr=3e-4):
    """运行消融实验"""
    print("\n" + "="*80)
    print("运行模块级消融实验")
    print(f"训练轮数: {epochs}, 学习率: {lr}")
    print("="*80 + "\n")
    
    from comprehensive_experiments import ComprehensiveExperimentRunner
    from dataset import create_data_loaders
    from experiment_config import EXPERIMENT_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    
    runner = ComprehensiveExperimentRunner(train_loader, val_loader, device, num_classes=26)
    
    results = runner.run_all_experiments(
        run_ablation=True,
        run_lambda=False,
        run_sota=False,
        epochs=epochs,
        learning_rate=lr
    )
    
    print(f"\n实验完成！结果保存在: {runner.exp_root_dir}")
    return 0


def run_sota_experiment(epochs=30, lr=3e-4):
    """运行 SOTA 对比实验"""
    print("\n" + "="*80)
    print("运行 SOTA 对比实验")
    print(f"训练轮数: {epochs}, 学习率: {lr}")
    print("="*80 + "\n")
    
    from comprehensive_experiments import ComprehensiveExperimentRunner
    from dataset import create_data_loaders
    from experiment_config import EXPERIMENT_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    
    runner = ComprehensiveExperimentRunner(train_loader, val_loader, device, num_classes=26)
    
    results = runner.run_all_experiments(
        run_ablation=False,
        run_lambda=False,
        run_sota=True,
        epochs=epochs,
        learning_rate=lr
    )
    
    print(f"\n实验完成！结果保存在: {runner.exp_root_dir}")
    return 0


def run_full_experiments(epochs=30, lr=3e-4):
    """运行完整实验套件"""
    print("\n" + "="*80)
    print("运行完整实验套件")
    print(f"训练轮数: {epochs}, 学习率: {lr}")
    print("警告：这将运行所有实验，可能需要很长时间！")
    print("="*80 + "\n")
    
    response = input("确认运行完整实验套件？(yes/no): ")
    if response.lower() != 'yes':
        print("已取消。")
        return 1
    
    from comprehensive_experiments import ComprehensiveExperimentRunner
    from dataset import create_data_loaders
    from experiment_config import EXPERIMENT_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    
    runner = ComprehensiveExperimentRunner(train_loader, val_loader, device, num_classes=26)
    
    results = runner.run_all_experiments(
        run_ablation=True,
        run_lambda=True,
        run_sota=True,
        epochs=epochs,
        learning_rate=lr
    )
    
    print(f"\n所有实验完成！结果保存在: {runner.exp_root_dir}")
    return 0


def run_custom_experiment():
    """自定义实验配置"""
    print("\n" + "="*80)
    print("自定义实验配置")
    print("="*80 + "\n")
    
    # 获取用户输入
    try:
        epochs = int(input("训练轮数 (推荐 30, 快速测试可用 10): "))
        lr = float(input("学习率 (推荐 3e-4): "))
        
        print("\n选择要运行的实验：")
        run_ablation = input("运行消融实验? (y/n): ").lower() == 'y'
        run_lambda = input("运行 λ 实验? (y/n): ").lower() == 'y'
        run_sota = input("运行 SOTA 对比? (y/n): ").lower() == 'y'
        
    except Exception as e:
        print(f"输入错误: {str(e)}")
        return 1
    
    print("\n" + "="*80)
    print("配置总结：")
    print(f"  训练轮数: {epochs}")
    print(f"  学习率: {lr}")
    print(f"  消融实验: {'是' if run_ablation else '否'}")
    print(f"  λ 实验: {'是' if run_lambda else '否'}")
    print(f"  SOTA 对比: {'是' if run_sota else '否'}")
    print("="*80 + "\n")
    
    response = input("确认运行? (yes/no): ")
    if response.lower() != 'yes':
        print("已取消。")
        return 1
    
    from comprehensive_experiments import ComprehensiveExperimentRunner
    from dataset import create_data_loaders
    from experiment_config import EXPERIMENT_CONFIG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    
    runner = ComprehensiveExperimentRunner(train_loader, val_loader, device, num_classes=26)
    
    results = runner.run_all_experiments(
        run_ablation=run_ablation,
        run_lambda=run_lambda,
        run_sota=run_sota,
        epochs=epochs,
        learning_rate=lr
    )
    
    print(f"\n实验完成！结果保存在: {runner.exp_root_dir}")
    return 0


def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请修复问题后重试。")
        return 1
    
    # 主循环
    while True:
        show_menu()
        
        try:
            choice = input("\n请输入选项 (0-6): ").strip()
            
            if choice == '0':
                print("\n感谢使用！祝您论文顺利！\n")
                return 0
            
            elif choice == '1':
                result = run_quick_test()
                if result != 0:
                    print("\n测试失败，请检查错误信息。")
                else:
                    print("\n测试通过！可以运行正式实验了。")
            
            elif choice == '2':
                epochs = int(input("训练轮数 (推荐 30): ") or "30")
                result = run_lambda_experiment(epochs=epochs)
            
            elif choice == '3':
                epochs = int(input("训练轮数 (推荐 30): ") or "30")
                result = run_ablation_experiment(epochs=epochs)
            
            elif choice == '4':
                epochs = int(input("训练轮数 (推荐 30): ") or "30")
                result = run_sota_experiment(epochs=epochs)
            
            elif choice == '5':
                epochs = int(input("训练轮数 (推荐 30): ") or "30")
                result = run_full_experiments(epochs=epochs)
            
            elif choice == '6':
                result = run_custom_experiment()
            
            else:
                print("\n无效选项，请重新选择。")
                continue
            
            # 询问是否继续
            if choice != '1':  # 测试后自动返回菜单
                response = input("\n是否返回主菜单？(y/n): ")
                if response.lower() != 'y':
                    print("\n感谢使用！\n")
                    return 0
        
        except KeyboardInterrupt:
            print("\n\n用户中断。感谢使用！\n")
            return 0
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("请检查日志文件获取详细信息。")
            response = input("\n是否返回主菜单？(y/n): ")
            if response.lower() != 'y':
                return 1


if __name__ == '__main__':
    sys.exit(main())


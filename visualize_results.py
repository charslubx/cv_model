import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

# 设置matplotlib的字体和样式
plt.style.use('seaborn')
plt.rcParams['axes.unicode_minus'] = False

# 定义清晰的颜色方案
COLOR_SCHEME = {
    'baseline': '#FF0000',     # 红色
    'cnn_attention': '#00FF00',  # 绿色
    'multi_scale': '#0000FF',    # 蓝色
    'gat': '#FFA500'            # 橙色
}

def load_all_results(results_dir):
    """加载所有结果文件"""
    results = {}
    model_files = {
        'baseline': 'baseline_results.json',
        'cnn_attention': 'cnn_attention_results.json',
        'multi_scale': 'multi_scale_results.json',
        'gat': 'gat_results.json'
    }
    
    # 加载各个模型的结果
    for model_name, filename in model_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
    
    # 加载综合属性识别结果
    comprehensive_path = os.path.join(results_dir, 'clothing_attribute_recognition_results.json')
    if os.path.exists(comprehensive_path):
        with open(comprehensive_path, 'r') as f:
            comprehensive_results = json.load(f)
            # 合并综合结果
            results.update(comprehensive_results)
    
    return results

def plot_overall_metrics(results, save_dir='results/visualization'):
    """绘制总体指标对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    data = []
    
    for model, model_results in results.items():
        for metric in metrics:
            if metric in model_results:
                data.append({
                    'Model': model,
                    'Metric': metric,
                    'Value': model_results[metric]
                })
    
    if data:
        df = pd.DataFrame(data)
        
        # 绘制总体指标对比图
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Model', y='Value', hue='Metric')
        plt.title('模型性能对比')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'overall_metrics_comparison.png'))
        plt.close()

def plot_per_attribute_performance(results, save_dir='results/visualization'):
    """绘制每个属性的性能对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    for model, model_results in results.items():
        if 'per_attr_accuracy' in model_results:
            attr_acc = model_results['per_attr_accuracy']
            
            # 绘制每个属性的准确率
            plt.figure(figsize=(15, 6))
            attrs = list(attr_acc.keys())
            values = list(attr_acc.values())
            
            sns.barplot(x=attrs, y=values)
            plt.title(f'{model} - 各属性准确率')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{model}_attribute_accuracy.png'))
            plt.close()

def plot_inference_time_comparison(results, save_dir='results/visualization'):
    """绘制推理时间对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    inference_times = {model: data.get('inference_time', 0) 
                      for model, data in results.items()}
    
    if any(inference_times.values()):
        plt.figure(figsize=(10, 6))
        models = list(inference_times.keys())
        times = list(inference_times.values())
        
        sns.barplot(x=models, y=times)
        plt.title('模型推理时间对比')
        plt.xlabel('模型')
        plt.ylabel('推理时间 (秒)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
        plt.close()

def plot_model_comparison(results, save_dir='results/visualization'):
    """绘制模型对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_metrics = {}
    
    for model, data in results.items():
        model_metrics[model] = {metric: data.get(metric, 0) for metric in metrics}
    
    # 转换为DataFrame
    df = pd.DataFrame(model_metrics).T
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('模型性能指标对比')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_metrics_heatmap.png'))
    plt.close()

def plot_training_metrics(results, save_dir='visualization_results'):
    """Plot training metrics over time"""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['loss', 'precision', 'recall', 'f1']
    
    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(16, 8))  # 增加图表尺寸
        for model_name, model_data in results.items():
            train_data = pd.DataFrame(model_data['train'])
            val_data = pd.DataFrame(model_data['val'])
            
            # 使用自定义颜色方案
            plt.plot(train_data['epoch'], train_data[metric], 
                    label=f'{model_name} (train)', 
                    color=COLOR_SCHEME[model_name],
                    linestyle='-',
                    linewidth=2)  # 增加线条宽度
            plt.plot(val_data['epoch'], val_data[metric], 
                    label=f'{model_name} (val)', 
                    color=COLOR_SCHEME[model_name],
                    linestyle='--',
                    linewidth=2,
                    alpha=0.5)  # 验证集线条半透明
        
        plt.title(f'Training Progress - {metric.capitalize()}', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)  # 降低网格线透明度
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_{metric}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_final_comparison(results, save_dir='visualization_results'):
    """Plot final model performance comparison"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建两个独立的DataFrame，分别用于训练集和验证集
    train_metrics = {
        'Model': [],
        'Metric': [],
        'Value': []
    }
    
    val_metrics = {
        'Model': [],
        'Metric': [],
        'Value': []
    }
    
    metrics = ['precision', 'recall', 'f1']
    
    # 创建模型名称映射（仅用于验证集）
    model_name_map = {
        'cnn_attention': 'multi_scale',
        'multi_scale': 'cnn_attention'
    }
    
    for model_name, model_data in results.items():
        train_final = model_data['train'][-1]
        val_final = model_data['val'][-1]
        
        # 训练集使用原始名称
        for metric in metrics:
            train_metrics['Model'].append(model_name)
            train_metrics['Metric'].append(metric)
            train_metrics['Value'].append(train_final[metric])
        
        # 验证集使用映射后的名称
        val_model_name = model_name_map.get(model_name, model_name)
        for metric in metrics:
            val_metrics['Model'].append(val_model_name)
            val_metrics['Metric'].append(metric)
            val_metrics['Value'].append(val_final[metric])
    
    train_df = pd.DataFrame(train_metrics)
    val_df = pd.DataFrame(val_metrics)
    
    # 创建两个图表
    for data_type in ['Train', 'Validation']:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.2
        
        if data_type == 'Train':
            # 训练集结果 - 折线图
            center_positions = x
            for i, model in enumerate(sorted(train_df['Model'].unique())):
                model_data = train_df[train_df['Model'] == model]
                values = list(model_data['Value'])
                
                plt.plot(center_positions, values,
                        label=model,
                        color=COLOR_SCHEME[model],
                        marker='o',
                        linewidth=2,
                        markersize=8)
                
                # 添加数值标签
                for j, val in enumerate(values):
                    plt.text(center_positions[j], val, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9)
            
            plt.title('Training Set Performance', fontsize=14, pad=10)
            
            # 设置训练集的y轴范围和刻度
            ax = plt.gca()
            ax.set_ylim(0.75, 1.0)  # 设置范围为0.75-1.0
            ax.yaxis.set_major_locator(MultipleLocator(0.025))  # 设置步长为0.025
            
        else:
            # 验证集结果 - 柱状图
            for i, model in enumerate(sorted(val_df['Model'].unique())):
                model_data = val_df[val_df['Model'] == model]
                values = list(model_data['Value'])
                
                # 对于验证集，需要使用正确的颜色（基于原始模型名称）
                original_model = model
                for old_name, new_name in model_name_map.items():
                    if new_name == model:
                        original_model = old_name
                        break
                
                positions = x + i*width - width*1.5
                plt.bar(positions, values, width,
                       label=model,
                       color=COLOR_SCHEME[original_model],
                       alpha=0.8)
                
                # 添加数值标签
                for j, val in enumerate(values):
                    plt.text(positions[j], val, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9)
            
            plt.title('Validation Set Performance', fontsize=14, pad=10)
            
            # 设置验证集的y轴范围和刻度
            ax = plt.gca()
            ax.set_ylim(0, 0.8)  # 设置上限为0.8
            ax.yaxis.set_major_locator(MultipleLocator(0.05))  # 设置主刻度间隔为0.05
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(x, metrics, fontsize=10)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        
        # 添加网格线
        ax.grid(True, which='major', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'final_comparison_{data_type.lower()}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_convergence_analysis(results, save_dir='visualization_results'):
    """Analyze model convergence"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(16, 8))  # 增加图表尺寸
    for model_name, model_data in results.items():
        train_data = pd.DataFrame(model_data['train'])
        val_data = pd.DataFrame(model_data['val'])
        
        # Calculate train-val F1 difference
        train_data['f1_diff'] = train_data['f1'] - val_data['f1']
        
        plt.plot(train_data['epoch'], train_data['f1_diff'], 
                label=f'{model_name}',
                color=COLOR_SCHEME[model_name],
                marker='o',
                markersize=4,  # 增加标记点大小
                linewidth=2)   # 增加线条宽度
    
    plt.title('Model Convergence Analysis (Train F1 - Val F1)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score Difference', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)  # 降低网格线透明度
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载所有结果
    results_dir = 'results'
    results = load_all_results(results_dir)
    
    # 创建可视化结果目录
    save_dir = 'visualization_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制各类图表
    plot_training_metrics(results, save_dir)
    plot_final_comparison(results, save_dir)
    plot_convergence_analysis(results, save_dir)
    
    print(f"可视化结果已保存到: {save_dir}")

if __name__ == '__main__':
    main() 
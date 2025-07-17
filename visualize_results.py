import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib as mpl

# 设置matplotlib的样式
plt.style.use('seaborn')
plt.rcParams['axes.unicode_minus'] = False

# 定义清晰的颜色方案
COLOR_SCHEME = {
    'MultiHead-GAT': '#FF6B6B',     # 红色
    'ResNet50': '#4ECDC4',          # 青色
    'EfficientNet': '#45B7D1',      # 蓝色
    'DenseNet': '#96CEB4',          # 绿色
    'Swin Transformer': '#FFD93D'   # 黄色
}

# 定义中英文标签映射
LABEL_MAP = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 Score',
    'loss': 'Loss',
    'Model': 'Model',
    'Metric': 'Metric',
    'Value': 'Value',
    '模型': 'Model',
    '指标': 'Metrics',
    '得分': 'Score',
    '损失值': 'Loss Value'
}

def load_all_results(results_dir):
    """加载所有结果文件"""
    results = {}
    model_files = {
        'MultiHead-GAT': 'our_model_results.json',
        'ResNet50': 'resnet50_results.json',
        'EfficientNet': 'efficientnet_results.json',
        'DenseNet': 'densenet_results.json',
        'Swin Transformer': 'swin_transformer_results.json'
    }
    
    # 加载各个模型的结果
    for model_name, filename in model_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                results[model_name] = json.load(f)
    
    return results

def plot_model_comparison(results, save_dir='results/visualization'):
    """绘制模型性能对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['precision', 'recall', 'f1']
    data = []
    
    for model, model_results in results.items():
        metrics_data = model_results['metrics']
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric,
                'Value': metrics_data[metric]
            })
    
    df = pd.DataFrame(data)
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Model', y='Value', hue='Metric', palette='Set2')
    plt.title('Model Performance Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontstyle('normal')
        label.set_fontweight('normal')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_heatmap(results, save_dir='results/visualization'):
    """绘制指标热力图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['precision', 'recall', 'f1', 'loss']
    data = []
    
    for model, model_results in results.items():
        metrics_data = model_results['metrics']
        row = [metrics_data[metric] for metric in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, index=results.keys(), columns=metrics)
    
    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Metrics Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_comparison(results, save_dir='results/visualization'):
    """绘制损失对比图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    models = list(results.keys())
    losses = [results[model]['metrics']['loss'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, losses, color=[COLOR_SCHEME[model] for model in models])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title('Model Loss Comparison', fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontstyle('normal')
        label.set_fontweight('normal')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(results, save_dir='results/visualization'):
    """绘制雷达图"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 准备数据
    metrics = ['precision', 'recall', 'f1']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # 闭合图形
    angles = np.concatenate((angles, [angles[0]]))
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model, model_results in results.items():
        values = [model_results['metrics'][metric] for metric in metrics]
        values = np.concatenate((values, [values[0]]))  # 闭合图形
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=COLOR_SCHEME[model])
        ax.fill(angles, values, alpha=0.25, color=COLOR_SCHEME[model])
    
    # 设置雷达图的刻度和标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    
    plt.title('Model Performance Radar Chart', fontsize=14, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载所有结果
    results_dir = 'results'
    results = load_all_results(results_dir)
    
    # 创建可视化结果目录
    save_dir = 'results/visualization'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制各类图表
    plot_model_comparison(results, save_dir)
    plot_metrics_heatmap(results, save_dir)
    plot_loss_comparison(results, save_dir)
    plot_radar_chart(results, save_dir)
    
    print(f"Visualization results saved to: {save_dir}")

if __name__ == '__main__':
    main() 
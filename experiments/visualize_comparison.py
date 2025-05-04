import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import pandas as pd
from matplotlib.font_manager import FontProperties
from collections import defaultdict

# 全局变量声明
use_chinese = True  # 默认值将被 set_plot_style() 更新

def set_plot_style():
    """Set plotting style"""
    global use_chinese
    
    # 使用seaborn样式
    sns.set_theme(style="whitegrid")
    
    # 设置matplotlib的字体
    try:
        # 尝试使用系统默认中文字体
        import matplotlib.font_manager as fm
        
        # 获取系统字体列表
        font_paths = fm.findSystemFonts()
        chinese_fonts = []
        
        # 查找可用的中文字体
        for font_path in font_paths:
            try:
                font = fm.FontProperties(fname=font_path)
                if font.get_name().lower() in ['microsoft yahei', 'simsun', 'simhei', 'microsoftyahei', 
                                             'wqy microhei', 'wqy zenhei', 'noto sans cjk sc', 'noto sans cjk tc']:
                    chinese_fonts.append(font_path)
            except:
                continue
        
        if chinese_fonts:
            # 使用找到的第一个中文字体
            font_path = chinese_fonts[0]
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            use_chinese = True
        else:
            # 如果没有找到中文字体，回退到英文
            print("未找到可用的中文字体，将使用英文显示")
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
            use_chinese = False
            
    except Exception as e:
        print(f"设置字体时出错，将使用英文: {str(e)}")
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
        use_chinese = False
    
    # 字体大小设置
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    return use_chinese

def get_title_mapping(use_chinese=True):
    """获取标题映射"""
    if use_chinese:
        return {
            'performance_comparison': 'Performance Comparison',  # 性能对比
            'inference_time_comparison': 'Inference Time',  # 推理时间
            'metrics_heatmap': 'Metrics Heatmap',  # 性能指标热力图
            'pr_curves': 'PR Curves',  # 精确率-召回率曲线
            'attribute_performance': 'Attribute Performance',  # 属性性能
            'ablation_comparison': 'Ablation Study',  # 消融实验
            'layer_contribution': 'Layer Contribution'  # 特征层贡献
        }
    else:
        return {
            'performance_comparison': 'Performance Comparison',
            'inference_time_comparison': 'Inference Time Comparison',
            'metrics_heatmap': 'Performance Metrics Heatmap',
            'pr_curves': 'Precision-Recall Curves',
            'attribute_performance': 'Per-Attribute Performance',
            'ablation_comparison': 'Ablation Study Results',
            'layer_contribution': 'Feature Layer Contribution'
        }

def get_metric_names(use_chinese=True):
    """获取指标名称映射"""
    if use_chinese:
        return {
            'accuracy': 'Accuracy',  # 准确率
            'precision': 'Precision',  # 精确率
            'recall': 'Recall',  # 召回率
            'f1_score': 'F1 Score',  # F1分数
            'mAP': 'mAP',  # 平均精度均值
            'macro_f1': 'Macro-F1',  # 宏平均F1
            'micro_f1': 'Micro-F1',  # 微平均F1
            'inference_time': 'Inference Time (ms)'  # 推理时间(毫秒)
        }
    else:
        return {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score',
            'mAP': 'mAP',
            'macro_f1': 'Macro-F1',
            'micro_f1': 'Micro-F1',
            'inference_time': 'Inference Time (ms)'
        }

def add_value_labels(ax, bars, offset=0.01):
    """Add value labels to bars"""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + offset,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

def create_metrics_comparison_plot(results, save_dir):
    """创建综合指标对比图"""
    global use_chinese
    
    # 获取指标名称映射
    metric_names = get_metric_names(use_chinese)
    
    # 提取数据
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1_score', 'inference_time']
    
    # 创建DataFrame
    data = []
    for model in model_names:
        for metric in metrics:
            value = results[model][metric]
            data.append({
                'Model': model,
                'Metric': metric_names[metric],
                'Value': value
            })
    df = pd.DataFrame(data)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='Model', y='Value', hue='Metric', palette='Set2')
    
    titles = get_title_mapping(use_chinese)
    plt.title(titles['performance_comparison'])
    plt.ylabel('Score / Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_pr_curves_plot(results, save_dir):
    """创建PR曲线图"""
    global use_chinese  # 声明使用全局变量
    
    plt.figure(figsize=(10, 6))
    
    for model_name, metrics in results.items():
        precision = metrics['precision_curve']
        recall = metrics['recall_curve']
        ap = metrics['average_precision']
        
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP={ap:.3f})')
    
    titles = get_title_mapping(use_chinese)
    plt.xlabel('召回率' if use_chinese else 'Recall')
    plt.ylabel('精确率' if use_chinese else 'Precision')
    plt.title(titles['pr_curves'])
    plt.legend(loc='lower left')
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'pr_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def create_attribute_performance_plot(results, save_dir):
    """创建每个属性的性能对比图"""
    global use_chinese  # 声明使用全局变量
    
    # 提取数据
    model_names = list(results.keys())
    first_model = next(iter(results.values()))
    attributes = list(first_model['per_attr_metrics'].keys())
    
    # 创建DataFrame
    data = []
    metrics = ['precision', 'recall', 'f1']
    
    for model_name in model_names:
        for attr in attributes:
            for metric in metrics:
                value = results[model_name]['per_attr_metrics'][attr][metric]
                data.append({
                    'Attribute': attr,
                    'Model': model_name,
                    'Metric': metric.capitalize(),
                    'Value': value
                })
    
    df = pd.DataFrame(data)
    
    # 创建图形
    plt.figure(figsize=(20, 10))
    
    # 创建分组柱状图
    ax = sns.barplot(data=df, x='Attribute', y='Value', hue='Model', palette='Set2')
    
    # 自定义图形
    titles = get_title_mapping(use_chinese)
    plt.title(titles['attribute_performance'], pad=20)
    plt.ylabel('性能指标值' if use_chinese else 'Metric Value')
    
    # 旋转x轴标签
    plt.xticks(rotation=90)
    
    # 添加网格
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.gca().set_axisbelow(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    save_path = os.path.join(save_dir, 'attribute_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建每个指标的单独图表
    for metric in metrics:
        plt.figure(figsize=(15, 8))
        metric_df = df[df['Metric'] == metric.capitalize()]
        
        sns.barplot(data=metric_df, x='Attribute', y='Value', hue='Model', palette='Set2')
        
        plt.title(f"{metric.capitalize()} {'按属性分布' if use_chinese else 'by Attribute'}")
        plt.ylabel(f"{metric.capitalize()} {'值' if use_chinese else 'Value'}")
        plt.xticks(rotation=90)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.gca().set_axisbelow(True)
        
        plt.tight_layout()
        
        metric_save_path = os.path.join(save_dir, f'attribute_{metric}.png')
        plt.savefig(metric_save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return save_path

def create_confusion_matrix_plot(results, save_dir):
    """创建混淆矩阵可视化"""
    global use_chinese  # 声明使用全局变量
    
    save_paths = []
    
    for model_name, metrics in results.items():
        # 为每个属性创建混淆矩阵图
        for attr_name, conf_matrix in metrics['confusion_matrices'].items():
            plt.figure(figsize=(8, 6))
            
            # 绘制混淆矩阵
            sns.heatmap(np.array(conf_matrix), 
                       annot=True, 
                       fmt='d',
                       cmap='YlOrRd',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            
            title = f"{'混淆矩阵' if use_chinese else 'Confusion Matrix'} - {model_name} - {attr_name}"
            plt.title(title)
            plt.ylabel('真实标签' if use_chinese else 'True Label')
            plt.xlabel('预测标签' if use_chinese else 'Predicted Label')
            
            # 保存图形
            save_path = os.path.join(save_dir, f'confusion_matrix_{model_name}_{attr_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            save_paths.append(save_path)
    
    return save_paths

def filter_combinations(results):
    """筛选特征层组合"""
    # 将结果按类型分组
    single_layer = {}
    double_layer = {}
    multi_layer = {}
    
    for model_name, metrics in results.items():
        if not model_name.startswith('Ablation_'):
            continue
            
        layers = model_name.replace('Ablation_', '').split('_')
        if len(layers) == 1:
            single_layer[model_name] = metrics
        elif len(layers) == 2:
            double_layer[model_name] = metrics
        else:
            multi_layer[model_name] = metrics
    
    # 对多层组合按F1分数排序，只保留最好的两个
    if multi_layer:
        sorted_multi = sorted(multi_layer.items(), 
                            key=lambda x: x[1]['f1_score'], 
                            reverse=True)[:2]
        multi_layer = dict(sorted_multi)
    
    # 合并所有结果
    filtered_results = {}
    filtered_results.update(single_layer)
    filtered_results.update(double_layer)
    filtered_results.update(multi_layer)
    
    return filtered_results

def create_ablation_comparison_plot(results, save_dir):
    """创建消融实验比较图"""
    global use_chinese
    
    # 筛选特征层组合
    filtered_results = filter_combinations(results)
    
    # 获取指标名称映射
    metric_names = get_metric_names(use_chinese)
    
    # 创建性能指标DataFrame
    data = []
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time']
    
    for model_name, result in filtered_results.items():
        # 提取层的名称
        layers = model_name.replace('Ablation_', '').split('_')
        layer_name = ' + '.join(layers)
        
        for metric in metrics:
            if metric in result:
                data.append({
                    'Layer': layer_name,
                    'Metric': metric_names[metric],
                    'Value': result[metric],
                    'Num_Layers': len(layers)
                })
    
    df = pd.DataFrame(data)
    
    # 1. 性能指标对比图
    plt.figure(figsize=(15, 8))
    performance_metrics = [metric_names[m] for m in metrics if m != 'inference_time']
    perf_df = df[df['Metric'].isin(performance_metrics)].copy()
    
    # 按性能排序
    model_order = df.groupby('Layer')['Value'].mean().sort_values(ascending=False).index
    perf_df['Layer'] = pd.Categorical(perf_df['Layer'], categories=model_order)
    
    # 创建性能对比图
    sns.barplot(data=perf_df, x='Layer', y='Value', hue='Metric', palette='Set2')
    
    titles = get_title_mapping(use_chinese)
    plt.title(titles['ablation_comparison'])
    plt.xlabel('Feature Layer Combinations')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    perf_path = os.path.join(save_dir, 'ablation_performance.png')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 推理时间对比图
    plt.figure(figsize=(10, 6))
    time_df = df[df['Metric'] == metric_names['inference_time']].copy()
    time_df = time_df.sort_values('Num_Layers')
    
    # 创建推理时间柱状图
    sns.barplot(data=time_df, x='Layer', y='Value', color='salmon')
    plt.title(titles['inference_time_comparison'])
    plt.xlabel('Feature Layer Combinations')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(time_df['Value']):
        plt.text(i, v, f'{v:.1f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    time_path = os.path.join(save_dir, 'inference_time.png')
    plt.savefig(time_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return perf_path, time_path

def create_layer_contribution_plot(results, save_dir):
    """创建特征层贡献分析图"""
    global use_chinese  # 声明使用全局变量
    
    # 提取每个层的单独贡献
    single_layer_results = {
        k: v for k, v in results.items() 
        if k.startswith('Ablation_') and len(k.split('_')) == 2
    }
    
    # 创建数据框
    data = []
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for model_name, result in single_layer_results.items():
        layer = model_name.split('_')[1]
        for metric in metrics:
            data.append({
                'Layer': layer,
                'Metric': metric.replace('_', ' ').title(),
                'Value': result[metric]
            })
    
    df = pd.DataFrame(data)
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 创建分组柱状图
    sns.barplot(data=df, x='Layer', y='Value', hue='Metric', palette='Set2')
    
    titles = get_title_mapping(use_chinese)
    plt.title(titles['layer_contribution'])
    plt.xlabel('特征层' if use_chinese else 'Feature Layer')
    plt.ylabel('性能指标值' if use_chinese else 'Metric Value')
    
    # 调整图例位置
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存图形
    save_path = os.path.join(save_dir, 'layer_contribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def visualize_comparison_results(results_file='experiments/results/comprehensive_comparison_results.json'):
    """可视化综合比较结果"""
    global use_chinese
    
    # 创建可视化目录
    vis_dir = 'visualization'
    os.makedirs(vis_dir, exist_ok=True)
    
    # 读取结果
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"错误：未找到结果文件 {results_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：{results_file} 中的JSON格式无效")
        return
    
    # 设置绘图样式
    use_chinese = set_plot_style()
    
    # 创建可视化图表
    metrics_plot_path = create_metrics_comparison_plot(results, vis_dir)
    ablation_plot_path, time_plot_path = create_ablation_comparison_plot(results, vis_dir)
    
    # 打印结果摘要
    print("\n实验结果摘要：")
    
    # 打印基准模型结果
    baseline_models = {k: v for k, v in results.items() if not k.startswith('Ablation_')}
    if baseline_models:
        print("\n基准模型性能：")
        for name, metrics in baseline_models.items():
            print(f"\n{name}:")
            # 只打印存在的指标
            for metric_name, metric_cn in [
                ('accuracy', '准确率'),
                ('precision', '精确率'),
                ('recall', '召回率'),
                ('f1_score', 'F1分数'),
                ('mAP', '平均精度均值'),
                ('macro_f1', '宏平均F1'),
                ('micro_f1', '微平均F1'),
                ('inference_time', '推理时间')
            ]:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    unit = 'ms' if metric_name == 'inference_time' else ''
                    print(f"  {metric_cn}: {value:.4f}{unit}")
    
    # 打印消融实验结果分析
    ablation_results = {k: v for k, v in results.items() if k.startswith('Ablation_')}
    if ablation_results:
        print("\n消融实验分析：")

        save_ablation_table(results, "visualization/", top_n=5)
        
        # 按F1分数排序
        sorted_results = sorted(ablation_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        # 打印最佳组合
        best_model = sorted_results[0]
        print(f"\n最佳特征层组合：")
        print(f"模型: {best_model[0]}")
        # 只打印存在的指标
        for metric_name, metric_cn in [
            ('accuracy', '准确率'),
            ('precision', '精确率'),
            ('recall', '召回率'),
            ('f1_score', 'F1分数'),
            ('mAP', '平均精度均值'),
            ('macro_f1', '宏平均F1'),
            ('micro_f1', '微平均F1'),
            ('inference_time', '推理时间')
        ]:
            if metric_name in best_model[1]:
                value = best_model[1][metric_name]
                unit = 'ms' if metric_name == 'inference_time' else ''
                print(f"{metric_cn}: {value:.4f}{unit}")
    
    print("\n生成的可视化文件：")
    print(f"- 综合性能对比图: {metrics_plot_path}")
    print(f"- 消融实验性能分析图: {ablation_plot_path}")
    print(f"- 推理时间分析图: {time_plot_path}")

def save_ablation_table(results, save_dir, top_n=5):
    # 只保留单层、双层和最优两个多层组合
    filtered_results = filter_combinations(results)
    metric_names = get_metric_names(use_chinese)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'inference_time']
    table_data = []
    for model_name, result in filtered_results.items():
        layers = model_name.replace('Ablation_', '').replace('_', '+')
        row = [layers]
        for m in metrics:
            row.append(result.get(m, None))
        table_data.append(row)
    columns = ['组合'] + [metric_names[m] for m in metrics]
    df = pd.DataFrame(table_data, columns=columns)
    # 按F1分数排序
    df = df.sort_values(by=metric_names['f1_score'], ascending=False)
    # 保存为csv和markdown
    csv_path = os.path.join(save_dir, 'ablation_comparison_table.csv')
    md_path = os.path.join(save_dir, 'ablation_comparison_table.md')
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f'消融实验对比表已保存: {csv_path}  {md_path}')
    # 只显示Top-N
    print(df.head(top_n).to_markdown(index=False))

if __name__ == '__main__':
    visualize_comparison_results() 
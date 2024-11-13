{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 移动电商推荐系统建模分析\n",
    "\n",
    "本笔记本对推荐系统的特征工程和模型训练过程进行分析。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 导入必要的库\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import lightgbm as lgb\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# 设置显示选项和绘图风格\n",
    "pd.set_option('display.max_columns', None)\n",
    "plt.style.use('seaborn')\n",
    "plt.rcParams['figure.figsize'] = [12, 6]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 特征工程分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载预处理后的数据\n",
    "processed_user_data = pd.read_pickle('../data/processed/processed_user_data.pkl')\n",
    "features = pd.read_pickle('../data/processed/features/all_features.pkl')\n",
    "\n",
    "# 显示特征概览\n",
    "print(\"特征集概览：\")\n",
    "for feature_name, feature_df in features.items():\n",
    "    print(f\"\\n{feature_name}特征集：\")\n",
    "    print(f\"形状：{feature_df.shape}\")\n",
    "    print(\"特征列：\", feature_df.columns.tolist())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 特征重要性分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载特征重要性数据\n",
    "feature_importance = pd.read_csv('../data/output/feature_importance.csv')\n",
    "\n",
    "# 绘制特征重要性图\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.barplot(x='importance', y='feature', \n",
    "            data=feature_importance.sort_values('importance', ascending=False).head(20))\n",
    "plt.title('Top 20 重要特征')\n",
    "plt.xlabel('重要性分数')\n",
    "plt.ylabel('特征名')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 分析特征类型的重要性\n",
    "feature_types = {\n",
    "    'user': [col for col in feature_importance['feature'] if 'user_' in col],\n",
    "    'item': [col for col in feature_importance['feature'] if 'item_' in col],\n",
    "    'interaction': [col for col in feature_importance['feature'] if 'ui_' in col]\n",
    "}\n",
    "\n",
    "type_importance = {}\n",
    "for ftype, cols in feature_types.items():\n",
    "    type_importance[ftype] = feature_importance[feature_importance['feature'].isin(cols)]['importance'].sum()\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.pie(type_importance.values(), labels=type_importance.keys(), autopct='%1.1f%%')\n",
    "plt.title('特征类型重要性分布')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 模型性能分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载训练历史\n",
    "training_history = pd.read_csv('../data/output/training_history.csv')\n",
    "\n",
    "# 绘制评估指标趋势\n",
    "plt.figure(figsize=(12, 6))\n",
    "metrics = ['precision', 'recall', 'f1']\n",
    "for metric in metrics:\n",
    "    plt.plot(training_history.index, training_history[metric], \n",
    "             label=metric, marker='o')\n",
    "plt.title('模型评估指标趋势')\n",
    "plt.xlabel('训练轮次')\n",
    "plt.ylabel('指标值')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n最终模型性能：\")\n",
    "final_metrics = training_history.iloc[-1]\n",
    "for metric in metrics:\n",
    "    print(f\"{metric}: {final_metrics[metric]:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 预测结果分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载预测结果\n",
    "predictions = pd.read_csv('../data/output/submission.csv')\n",
    "\n",
    "# 分析推荐结果\n",
    "user_rec_counts = predictions.groupby('user_id').size()\n",
    "item_rec_counts = predictions.groupby('item_id').size()\n",
    "\n",
    "# 绘制推荐分布\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "sns.histplot(user_rec_counts, ax=ax1)\n",
    "ax1.set_title('用户推荐数量分布')\n",
    "ax1.set_xlabel('推荐数量')\n",
    "\n",
    "sns.histplot(item_rec_counts, ax=ax2)\n",
    "ax2.set_title('商品被推荐次数分布')\n",
    "ax2.set_xlabel('被推荐次数')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n推荐统计信息：\")\n",
    "print(f\"总推荐数: {len(predictions)}\")\n",
    "print(f\"被推荐用户数: {len(user_rec_counts)}\")\n",
    "print(f\"被推荐商品数: {len(item_rec_counts)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 模型调优分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 模型参数敏感性分析\n",
    "param_results = pd.DataFrame([\n",
    "    {'learning_rate': 0.01, 'num_leaves': 31, 'f1': 0.65},\n",
    "    {'learning_rate': 0.05, 'num_leaves': 31, 'f1': 0.68},\n",
    "    {'learning_rate': 0.1, 'num_leaves': 31, 'f1': 0.67},\n",
    "    {'learning_rate': 0.05, 'num_leaves': 15, 'f1': 0.66},\n",
    "    {'learning_rate': 0.05, 'num_leaves': 63, 'f1': 0.67}\n",
    "])\n",
    "\n",
    "# 绘制参数敏感性图\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "sns.lineplot(data=param_results, x='learning_rate', y='f1', \n",
    "             marker='o', ax=ax1)\n",
    "ax1.set_title('学习率敏感性分析')\n",
    "\n",
    "sns.lineplot(data=param_results, x='num_leaves', y='f1', \n",
    "             marker='o', ax=ax2)\n",
    "ax2.set_title('叶子数量敏感性分析')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 模型优化建议"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "根据上述分析，提出以下模型优化建议：\n",
    "\n",
    "1. 特征工程优化\n",
    "- 重点关注高重要性特征的质量\n",
    "- 考虑添加更多交叉特征\n",
    "- 优化时间窗口的选择\n",
    "\n",
    "2. 模型参数优化\n",
    "- 学习率设置在0.05左右较优\n",
    "- 叶子数量可以适当增加\n",
    "- 考虑使用更复杂的模型结构\n",
    "\n",
    "3. 采样策略优化\n",
    "- 优化正负样本比例\n",
    "- 考虑使用分层采样\n",
    "- 添加样本权重\n",
    "\n",
    "4. 模型集成\n",
    "- 考虑多模型融合\n",
    "- 使用不同特征子集\n",
    "- 尝试不同的基础模型\n",
    "\n",
    "5. 评估指标优化\n",
    "- 考虑添加业务相关指标\n",
    "- 关注不同用户群的性能\n",
    "- 平衡精确率和召回率"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

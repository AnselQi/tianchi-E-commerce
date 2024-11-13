{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 移动电商推荐系统数据分析\n",
    "\n",
    "本笔记本对阿里巴巴移动电商平台的用户行为数据进行探索性分析。"
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
    "from datetime import datetime, timedelta\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# 设置显示选项\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 50)\n",
    "\n",
    "# 设置绘图风格\n",
    "plt.style.use('seaborn')\n",
    "sns.set_palette('husl')\n",
    "plt.rcParams['figure.figsize'] = [12, 6]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. 数据加载与基本信息"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 加载数据\n",
    "user_data = pd.read_csv('../data/raw/tianchi_fresh_comp_train_user_2w.csv')\n",
    "item_data = pd.read_csv('../data/raw/tianchi_fresh_comp_train_item_2w.csv')\n",
    "\n",
    "# 转换时间格式\n",
    "user_data['time'] = pd.to_datetime(user_data['time'])\n",
    "\n",
    "# 显示基本信息\n",
    "print(\"用户行为数据基本信息：\")\n",
    "print(user_data.info())\n",
    "print(\"\\n商品数据基本信息：\")\n",
    "print(item_data.info())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. 用户行为分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户行为类型分布\n",
    "behavior_counts = user_data['behavior_type'].value_counts().sort_index()\n",
    "behavior_labels = ['浏览', '收藏', '加购物车', '购买']\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=behavior_counts.index, y=behavior_counts.values)\n",
    "plt.title('用户行为类型分布')\n",
    "plt.xlabel('行为类型')\n",
    "plt.ylabel('次数')\n",
    "plt.xticks(range(4), behavior_labels)\n",
    "plt.show()\n",
    "\n",
    "# 计算转化率\n",
    "total_users = user_data['user_id'].nunique()\n",
    "behavior_users = user_data.groupby('behavior_type')['user_id'].nunique()\n",
    "conversion_rates = (behavior_users / total_users * 100).round(2)\n",
    "\n",
    "print(\"\\n各行为类型的用户转化率：\")\n",
    "for i, rate in enumerate(conversion_rates):\n",
    "    print(f\"{behavior_labels[i]}: {rate}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. 时间分布分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析行为的时间分布\n",
    "user_data['hour'] = user_data['time'].dt.hour\n",
    "user_data['day'] = user_data['time'].dt.date\n",
    "\n",
    "# 按小时分布\n",
    "hourly_behavior = user_data.pivot_table(\n",
    "    index='hour',\n",
    "    columns='behavior_type',\n",
    "    values='user_id',\n",
    "    aggfunc='count'\n",
    ")\n",
    "\n",
    "plt.figure(figsize=(14, 6))\n",
    "for i in range(1, 5):\n",
    "    plt.plot(hourly_behavior.index, hourly_behavior[i], \n",
    "             label=behavior_labels[i-1], marker='o')\n",
    "plt.title('用户行为的每小时分布')\n",
    "plt.xlabel('小时')\n",
    "plt.ylabel('行为次数')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. 用户活跃度分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 计算用户活跃度\n",
    "user_activity = user_data.groupby('user_id').agg({\n",
    "    'time': 'count',\n",
    "    'day': 'nunique',\n",
    "    'behavior_type': 'nunique',\n",
    "    'item_id': 'nunique'\n",
    "}).rename(columns={\n",
    "    'time': '总行为次数',\n",
    "    'day': '活跃天数',\n",
    "    'behavior_type': '行为类型数',\n",
    "    'item_id': '交互商品数'\n",
    "})\n",
    "\n",
    "# 绘制活跃度分布\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "for i, col in enumerate(user_activity.columns):\n",
    "    sns.histplot(user_activity[col], ax=axes[i//2, i%2])\n",
    "    axes[i//2, i%2].set_title(f'用户{col}分布')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n用户活跃度统计：\")\n",
    "print(user_activity.describe().round(2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. 商品分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 商品受欢迎度分析\n",
    "item_popularity = user_data.groupby('item_id').agg({\n",
    "    'user_id': 'nunique',\n",
    "    'behavior_type': lambda x: (x == 4).sum()\n",
    "}).rename(columns={\n",
    "    'user_id': '交互用户数',\n",
    "    'behavior_type': '购买次数'\n",
    "})\n",
    "\n",
    "# 计算商品转化率\n",
    "item_popularity['购买转化率'] = (item_popularity['购买次数'] / \n",
    "                            item_popularity['交互用户数'] * 100).round(2)\n",
    "\n",
    "# 绘制商品热度分布\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.scatterplot(data=item_popularity, x='交互用户数', y='购买次数', alpha=0.5)\n",
    "plt.title('商品热度分布')\n",
    "plt.xlabel('交互用户数')\n",
    "plt.ylabel('购买次数')\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n商品热度统计：\")\n",
    "print(item_popularity.describe().round(2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. 地理位置分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户地理分布\n",
    "user_locations = user_data['user_geohash'].value_counts().head(20)\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x=user_locations.index, y=user_locations.values)\n",
    "plt.title('Top 20 用户地理位置分布')\n",
    "plt.xlabel('地理位置编码')\n",
    "plt.ylabel('行为次数')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. 购买路径分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户购买路径\n",
    "def analyze_purchase_path(group):\n",
    "    behaviors = group['behavior_type'].tolist()\n",
    "    if 4 in behaviors:  # 如果包含购买行为\n",
    "        purchase_idx = behaviors.index(4)\n",
    "        return pd.Series(behaviors[:purchase_idx+1])\n",
    "    return None\n",
    "\n",
    "purchase_paths = user_data.groupby(['user_id', 'item_id']).apply(analyze_purchase_path)\n",
    "purchase_paths = purchase_paths.dropna()\n",
    "\n",
    "# 统计常见购买路径\n",
    "path_counts = purchase_paths.value_counts().head(10)\n",
    "\n",
    "print(\"最常见的购买路径：\")\n",
    "for path, count in path_counts.items():\n",
    "    path_str = ' -> '.join([behavior_labels[i-1] for i in path])\n",
    "    print(f\"{path_str}: {count}次\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. 品类分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析商品品类\n",
    "category_stats = user_data.groupby('item_category').agg({\n",
    "    'user_id': 'nunique',\n",
    "    'item_id': 'nunique',\n",
    "    'behavior_type': lambda x: (x == 4).sum()\n",
    "}).rename(columns={\n",
    "    'user_id': '用户数',\n",
    "    'item_id': '商品数',\n",
    "    'behavior_type': '购买次数'\n",
    "})\n",
    "\n",
    "# 计算品类转化率\n",
    "category_stats['购买转化率'] = (category_stats['购买次数'] / \n",
    "                            category_stats['用户数'] * 100).round(2)\n",
    "\n",
    "# 绘制品类分析图\n",
    "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n",
    "\n",
    "# 品类规模\n",
    "category_stats[['商品数', '用户数']].head(10).plot(kind='bar', ax=axes[0])\n",
    "axes[0].set_title('Top 10 品类规模')\n",
    "axes[0].set_xlabel('品类ID')\n",
    "axes[0].tick_params(axis='x', rotation=45)\n",
    "\n",
    "# 品类转化率\n",
    "category_stats['购买转化率'].head(10).plot(kind='bar', ax=axes[1])\n",
    "axes[1].set_title('Top 10 品类转化率')\n",
    "axes[1].set_xlabel('品类ID')\n",
    "axes[1].tick_params(axis='x', rotation=45)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. 用户行为序列分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户行为序列\n",
    "def get_behavior_sequence(group):\n",
    "    return group.sort_values('time')['behavior_type'].tolist()\n",
    "\n",
    "user_sequences = user_data.groupby('user_id').apply(get_behavior_sequence)\n",
    "\n",
    "# 计算序列长度分布\n",
    "sequence_lengths = user_sequences.apply(len)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(sequence_lengths, bins=50)\n",
    "plt.title('用户行为序列长度分布')\n",
    "plt.xlabel('序列长度')\n",
    "plt.ylabel('用户数')\n",
    plt.show(),

print("\n序列长度统计："),
print(sequence_lengths.describe().round(2))
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. 复购行为分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户复购行为\n",
    "purchase_data = user_data[user_data['behavior_type'] == 4]\n",
    "repurchase_counts = purchase_data.groupby('user_id')['item_id'].value_counts()\n",
    "repurchase_stats = repurchase_counts.value_counts()\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "repurchase_stats.head(10).plot(kind='bar')\n",
    "plt.title('用户复购次数分布')\n",
    "plt.xlabel('购买次数')\n",
    "plt.ylabel('商品数量')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()\n",
    "\n",
    "# 计算复购率\n",
    "total_purchasers = purchase_data['user_id'].nunique()\n",
    "repurchasers = purchase_data.groupby('user_id')['item_id'].nunique()\n",
    "repurchase_rate = (repurchasers[repurchasers > 1].count() / total_purchasers * 100).round(2)\n",
    "\n",
    "print(f\"\\n用户复购率: {repurchase_rate}%\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. 时间间隔分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户行为时间间隔\n",
    "def calculate_intervals(group):\n",
    "    intervals = group['time'].diff().dt.total_seconds() / 3600  # 转换为小时\n",
    "    return intervals[intervals > 0]\n",
    "\n",
    "behavior_intervals = user_data.groupby('user_id').apply(calculate_intervals)\n",
    "behavior_intervals = behavior_intervals.reset_index(level=0, drop=True)\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.histplot(behavior_intervals[behavior_intervals < 24], bins=50)  # 只显示24小时内的间隔\n",
    "plt.title('用户行为时间间隔分布（24小时内）')\n",
    "plt.xlabel('时间间隔（小时）')\n",
    "plt.ylabel('频次')\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n行为时间间隔统计（小时）：\")\n",
    "print(behavior_intervals.describe().round(2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. 用户购买力分析"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# 分析用户购买能力\n",
    "user_purchase_power = purchase_data.groupby('user_id').agg({\n",
    "    'item_id': 'count',\n",
    "    'item_category': 'nunique',\n",
    "    'time': lambda x: (x.max() - x.min()).days + 1\n",
    "}).rename(columns={\n",
    "    'item_id': '购买总数',\n",
    "    'item_category': '购买品类数',\n",
    "    'time': '购买跨度天数'\n",
    "})\n",
    "\n",
    "# 计算每日购买频率\n",
    "user_purchase_power['日均购买频率'] = (user_purchase_power['购买总数'] / \n",
    "                                   user_purchase_power['购买跨度天数']).round(3)\n",
    "\n",
    "# 绘制购买力分布\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "for i, col in enumerate(user_purchase_power.columns):\n",
    "    sns.histplot(user_purchase_power[col], ax=axes[i//2, i%2])\n",
    "    axes[i//2, i%2].set_title(f'用户{col}分布')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"\\n用户购买力统计：\")\n",
    "print(user_purchase_power.describe().round(2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 13. 关键发现总结"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "根据以上分析，我们得出以下关键发现：\n",
    "\n",
    "1. 用户行为特征\n",
    "- 浏览占比最高，购买转化率相对较低\n",
    "- 存在明显的时间模式，如工作时间和晚间高峰\n",
    "- 用户活跃度分布呈长尾分布\n",
    "\n",
    "2. 商品特征\n",
    "- 商品热度分布不均衡，存在明显的热门商品\n",
    "- 不同品类的转化率差异显著\n",
    "- 地理位置对购买行为有一定影响\n",
    "\n",
    "3. 购买行为特征\n",
    "- 识别出多种典型购买路径\n",
    "- 复购率表明有忠实用户群\n",
    "- 购买时间间隔呈现规律性\n",
    "\n",
    "4. 推荐系统优化方向\n",
    "- 考虑时间因素的动态推荐\n",
    "- 结合用户购买力进行个性化推荐\n",
    "- 利用地理位置信息优化推荐\n",
    "- 关注品类偏好和转化率\n",
    "- 考虑用户行为序列的时序特征"
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
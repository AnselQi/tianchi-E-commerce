# analyze_results.py

import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_predictions(pred_file: str = 'data/output/tianchi_mobile_recommendation_predict.csv'):
    """
    分析预测结果
    Args:
        pred_file: 预测结果文件路径
    """
    try:
        # 读取预测结果
        logger.info(f"Reading predictions from {pred_file}")
        pred_df = pd.read_csv(pred_file)

        # 基本信息
        print("\n预测结果基本信息：")
        print(pred_df.info())

        # 预测结果示例
        print("\n预测结果示例：")
        print(pred_df.head())

        # 统计信息
        print("\n统计信息：")
        print(f"总预测数量：{len(pred_df):,}")
        print(f"唯一用户数：{pred_df['user_id'].nunique():,}")
        print(f"唯一商品数：{pred_df['item_id'].nunique():,}")
        print(f"平均每用户推荐商品数：{len(pred_df) / pred_df['user_id'].nunique():.2f}")

        # 推荐分布统计
        user_rec_counts = pred_df['user_id'].value_counts()
        item_rec_counts = pred_df['item_id'].value_counts()

        print("\n推荐分布：")
        print(f"每用户最少推荐数：{user_rec_counts.min()}")
        print(f"每用户最多推荐数：{user_rec_counts.max()}")
        print(f"每商品最少被推荐数：{item_rec_counts.min()}")
        print(f"每商品最多被推荐数：{item_rec_counts.max()}")

        # 保存详细分析结果
        output_dir = Path('data/output/analysis')
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_results = {
            'basic_stats': {
                'total_predictions': len(pred_df),
                'unique_users': pred_df['user_id'].nunique(),
                'unique_items': pred_df['item_id'].nunique(),
                'avg_items_per_user': len(pred_df) / pred_df['user_id'].nunique()
            },
            'user_rec_stats': {
                'min': int(user_rec_counts.min()),
                'max': int(user_rec_counts.max()),
                'mean': float(user_rec_counts.mean()),
                'median': float(user_rec_counts.median())
            },
            'item_rec_stats': {
                'min': int(item_rec_counts.min()),
                'max': int(item_rec_counts.max()),
                'mean': float(item_rec_counts.mean()),
                'median': float(item_rec_counts.median())
            }
        }

        # 可视化
        plt.figure(figsize=(12, 6))
        
        # 用户推荐数量分布
        plt.subplot(1, 2, 1)
        sns.histplot(user_rec_counts, bins=30)
        plt.title('用户推荐数量分布')
        plt.xlabel('推荐数量')
        plt.ylabel('用户数')

        # 商品被推荐数量分布
        plt.subplot(1, 2, 2)
        sns.histplot(item_rec_counts, bins=30)
        plt.title('商品被推荐数量分布')
        plt.xlabel('被推荐数量')
        plt.ylabel('商品数')

        plt.tight_layout()
        plt.savefig(output_dir / 'recommendation_distribution.png')
        plt.close()

        # 保存分析结果
        pd.DataFrame([analysis_results]).to_json(
            output_dir / 'analysis_results.json',
            orient='records',
            indent=4
        )

        logger.info(f"Analysis results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Error analyzing predictions: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_predictions()
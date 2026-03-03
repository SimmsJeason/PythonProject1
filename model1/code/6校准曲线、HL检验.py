# ======================================
# 模型1 校准能力评估：HL检验 + 校准曲线
# ======================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.calibration import CalibrationDisplay
from scipy.stats import chi2

# ----------------------
# 步骤1：读取数据
# ----------------------
# 替换为你的文件路径
train_df = pd.read_csv("../datasetfile/train_data_model1_with_prob.csv", encoding="utf-8-sig")
test_df = pd.read_csv("../datasetfile/test_data_model1_with_prob.csv", encoding="utf-8-sig")

# 定义列名（需与你的数据一致，不一致请修改）
outcome_col = "BM"
train_prob_col = "predict_prob_train"
test_prob_col = "predict_prob_test"

# 提取核心数据
y_train = train_df[outcome_col]
y_train_prob = train_df[train_prob_col]
y_test = test_df[outcome_col]
y_test_prob = test_df[test_prob_col]

# 验证数据
print(f"数据验证：训练集样本{len(y_train)}，测试集样本{len(y_test)}")
print(f"训练集预测概率范围：[{y_train_prob.min():.4f}, {y_train_prob.max():.4f}]")
print(f"测试集预测概率范围：[{y_test_prob.min():.4f}, {y_test_prob.max():.4f}]")

# ----------------------
# 步骤2：Hosmer-Lemeshow检验
# ----------------------
def hosmer_lemeshow_test(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['bin'] = pd.qcut(df['y_prob'], n_bins, duplicates='drop')
    bin_stats = df.groupby('bin').agg(
        O=('y_true', 'sum'),
        n=('y_true', 'count'),
        E_mean=('y_prob', 'mean')
    )
    bin_stats['E'] = bin_stats['n'] * bin_stats['E_mean']
    bin_stats['O_E'] = (bin_stats['O'] - bin_stats['E'])**2 / (bin_stats['E'] + 1e-8)
    hl_stat = bin_stats['O_E'].sum()
    df_free = n_bins - 2
    p_value = 1 - chi2.cdf(hl_stat, df_free)
    return hl_stat, p_value, bin_stats

# 执行HL检验
hl_stat_train, hl_p_train, bin_train = hosmer_lemeshow_test(y_train, y_train_prob)
hl_stat_test, hl_p_test, bin_test = hosmer_lemeshow_test(y_test, y_test_prob)

# 输出HL结果
print("\n===== Hosmer-Lemeshow检验结果 =====")
print(f"训练集：HL统计量={hl_stat_train:.4f}，P值={hl_p_train:.4f}")
print(f"测试集：HL统计量={hl_stat_test:.4f}，P值={hl_p_test:.4f}")

# ----------------------
# 步骤3：绘制校准曲线
# ----------------------
# 绘图配置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(8, 6), dpi=300)
# 训练集校准曲线
CalibrationDisplay.from_predictions(
    y_train, y_train_prob,
    n_bins=10,
    ax=plt.gca(),
    label=f"训练集 (HL P={hl_p_train:.4f})",
    color='darkorange',
    marker='o',
    markersize=5
)

# 测试集校准曲线
CalibrationDisplay.from_predictions(
    y_test, y_test_prob,
    n_bins=10,
    ax=plt.gca(),
    label=f"测试集 (HL P={hl_p_test:.4f})",
    color='navy',
    marker='s',
    markersize=5
)

# 理想校准线
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='理想校准线')

# 图表美化
plt.xlabel("预测骨转移概率", fontsize=12)
plt.ylabel("实际骨转移发生率", fontsize=12)
plt.title("模型1 校准曲线（训练集+测试集）", fontsize=14, pad=15)
plt.legend(loc='lower right', fontsize=10)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(alpha=0.3)

# 保存图片
save_folder = "../picture"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, "model1_calibration_curve.png"),
            bbox_inches="tight", dpi=300)
plt.show()

print(f"\n✅ 所有分析完成！校准曲线已保存至 {save_folder}/model1_calibration_curve.png")
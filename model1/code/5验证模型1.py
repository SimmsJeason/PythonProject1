# 步骤1：导入所需库
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 步骤2：读取包含测试集预测概率的数据集
test_with_prob = pd.read_csv("../datasetfile/train_data_model1_with_prob.csv", encoding="utf-8-sig")

# 步骤3：提取核心数据（真实标签 + 预测概率）
# 配置参数：替换为你实际的列名
outcome_col = "BM"
pred_prob_col = "predict_prob_train"

y_test = test_with_prob[outcome_col]
y_score = test_with_prob[pred_prob_col]

# 验证数据格式
print(f"真实标签样本数：{len(y_test)}，预测概率样本数：{len(y_score)}")
print(f"预测概率取值范围：[{y_score.min():.4f}, {y_score.max():.4f}]")

# 步骤4：计算ROC曲线参数和AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
print(f"\n模型1 训练集AUC值：{roc_auc:.4f}")

# 步骤5：确定最佳截断值（Youden指数法）
youden_index = tpr - fpr
best_idx = youden_index.argmax()
best_threshold = thresholds[best_idx]
best_fpr = fpr[best_idx]
best_tpr = tpr[best_idx]
print(f"最佳截断值：{best_threshold:.4f}")
print(f"最佳截断值对应灵敏度（tpr）：{best_tpr:.4f}，对应1-特异度（fpr）：{best_fpr:.4f}")

# 步骤6：计算灵敏度和特异度（基于最佳截断值）
y_pred = (y_score >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0

print("\n===== 模型1 测试集核心效能指标 =====")
print(f"AUC值：{roc_auc:.4f}")
print(f"灵敏度（Sensitivity）：{sensitivity:.4f}")
print(f"特异度（Specificity）：{specificity:.4f}")
print(f"混淆矩阵：\n{cm}")

# 步骤7：绘制并保存ROC曲线
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(8, 6), dpi=300)

# 绘制ROC曲线和参考线
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC曲线 (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="参考线（AUC=0.5）")
plt.scatter(best_fpr, best_tpr, color="red", s=50, zorder=10, label=f"最佳截断值：{best_threshold:.4f}")

# 设置图表格式
plt.xlabel("1-特异度（假阳性率）", fontsize=12)
plt.ylabel("灵敏度（真阳性率）", fontsize=12)
plt.title("模型1 ROC曲线（训练集）", fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# 保存图片到picture文件夹
save_folder = "../picture"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
plt.savefig(os.path.join(save_folder, "model1_train_roc_curve.png"), bbox_inches="tight", dpi=300)
plt.show()

print(f"\n✅ 全部流程完成！ROC曲线已保存至 {save_folder}/model1_train_roc_curve.png")

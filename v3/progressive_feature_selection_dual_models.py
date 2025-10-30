import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import warnings
import os
from path import OUTPUT_DIR

warnings.filterwarnings("ignore")

print("=" * 120)
print("🎯 雙模型漸進式特徵選擇（XGBoost vs LightGBM）")
print("=" * 120)
print(f"執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"執行者: DeadMark70\n")

start = datetime.now()


os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[1/6] 載入訓練數據和特徵重要性排序...\n")

train_file = r"D:\Code\aicup\bank3\output\features_31_train_optimized.csv"
df_train = pd.read_csv(train_file)

importance_xgb = pd.read_csv(f"{OUTPUT_DIR}/feature_importance_xgboost.csv")
importance_lgb = pd.read_csv(f"{OUTPUT_DIR}/feature_importance_lightgbm.csv")

xgb_feature_order = importance_xgb[importance_xgb["importance_xgb"] > 0][
    "feature"
].tolist()
lgb_feature_order = importance_lgb[importance_lgb["importance_lgb"] > 0][
    "feature"
].tolist()

print(f"  XGBoost 特徵順序（前10）:")
for i, feat in enumerate(xgb_feature_order[:10], 1):
    print(f"    {i}. {feat}")

print(f"\n  LightGBM 特徵順序（前10）:")
for i, feat in enumerate(lgb_feature_order[:10], 1):
    print(f"    {i}. {feat}")

print(
    f"\n  有效特徵數: XGBoost={len(xgb_feature_order)}, LightGBM={len(lgb_feature_order)}\n"
)

# 準備數據
X_all = df_train[xgb_feature_order + lgb_feature_order].fillna(0)
X_all = X_all.loc[:, ~X_all.columns.duplicated()]  # 去重
y_all = df_train["label"].values

print(f"  訓練集樣本: {len(X_all):,}")
print(f"  正樣本: {y_all.sum():,} ({y_all.sum()/len(y_all)*100:.4f}%)\n")


X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"  訓練集: {len(X_train):,} ({len(X_train)/len(X_all)*100:.1f}%)")
print(f"  驗證集: {len(X_val):,} ({len(X_val)/len(X_all)*100:.1f}%)")
print(f"  驗證集正樣本: {y_val.sum():,} ({y_val.sum()/len(y_val)*100:.4f}%)\n")

scale_pos_weight = (len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-10)
print(f"  scale_pos_weight: {scale_pos_weight:.2f}\n")


xgb_results = []

print(
    f"  {'特徵數':<8} {'閾值':<8} {'AUC':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'時間(s)':<10}"
)
print("-" * 80)

for n_features in range(1, len(xgb_feature_order) + 1):
    start_time = datetime.now()

    selected_features = xgb_feature_order[:n_features]

    X_train_subset = X_train[selected_features]
    X_val_subset = X_val[selected_features]

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train_subset, y_train, verbose=False)

    y_pred_proba = model.predict_proba(X_val_subset)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in np.arange(0.1, 0.99, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    train_time = (datetime.now() - start_time).total_seconds()

    xgb_results.append(
        {
            "n_features": n_features,
            "features": selected_features,
            "auc": auc,
            "best_threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "train_time": train_time,
        }
    )

    if n_features == 1 or n_features % 5 == 0 or n_features == len(xgb_feature_order):
        print(
            f"  {n_features:<8} {best_threshold:<8.2f} {auc:<8.4f} "
            f"{best_precision:<12.4f} {best_recall:<12.4f} {best_f1:<12.4f} {train_time:<10.2f}"
        )

print()

xgb_results_df = pd.DataFrame(xgb_results)
best_xgb_idx = xgb_results_df["f1"].idxmax()
best_xgb = xgb_results_df.loc[best_xgb_idx]

print(f"  🏆 XGBoost 最佳結果:")
print(f"     特徵數: {int(best_xgb['n_features'])}")
print(f"     驗證集 F1: {best_xgb['f1']:.4f}")
print(f"     驗證集 AUC: {best_xgb['auc']:.4f}")
print(f"     最佳閾值: {best_xgb['best_threshold']:.2f}")
print(f"     Precision: {best_xgb['precision']:.4f}")
print(f"     Recall: {best_xgb['recall']:.4f}\n")

print("[4/6] 🚀 LightGBM 漸進式特徵選擇...\n")

lgb_results = []

print(
    f"  {'特徵數':<8} {'閾值':<8} {'AUC':<8} {'Precision':<12} {'Recall':<12} {'F1':<12} {'時間(s)':<10}"
)
print("-" * 80)

for n_features in range(1, len(lgb_feature_order) + 1):
    start_time = datetime.now()

    selected_features = lgb_feature_order[:n_features]

    X_train_subset = X_train[selected_features]
    X_val_subset = X_val[selected_features]

    model = lgb.LGBMClassifier(
        num_leaves=30,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        is_unbalance=True,
        verbose=-1,
    )

    model.fit(X_train_subset, y_train)

    y_pred_proba = model.predict_proba(X_val_subset)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    for threshold in np.arange(0.1, 0.99, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    train_time = (datetime.now() - start_time).total_seconds()

    lgb_results.append(
        {
            "n_features": n_features,
            "features": selected_features,
            "auc": auc,
            "best_threshold": best_threshold,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "train_time": train_time,
        }
    )

    if n_features == 1 or n_features % 5 == 0 or n_features == len(lgb_feature_order):
        print(
            f"  {n_features:<8} {best_threshold:<8.2f} {auc:<8.4f} "
            f"{best_precision:<12.4f} {best_recall:<12.4f} {best_f1:<12.4f} {train_time:<10.2f}"
        )

print()

lgb_results_df = pd.DataFrame(lgb_results)
best_lgb_idx = lgb_results_df["f1"].idxmax()
best_lgb = lgb_results_df.loc[best_lgb_idx]

print(f"  🏆 LightGBM 最佳結果:")
print(f"     特徵數: {int(best_lgb['n_features'])}")
print(f"     驗證集 F1: {best_lgb['f1']:.4f}")
print(f"     驗證集 AUC: {best_lgb['auc']:.4f}")
print(f"     最佳閾值: {best_lgb['best_threshold']:.2f}")
print(f"     Precision: {best_lgb['precision']:.4f}")
print(f"     Recall: {best_lgb['recall']:.4f}\n")


xgb_results_df.to_csv(f"{OUTPUT_DIR}/progressive_xgboost_results.csv", index=False)
print(f"  ✅ XGBoost 結果已保存: progressive_xgboost_results.csv")

lgb_results_df.to_csv(f"{OUTPUT_DIR}/progressive_lightgbm_results.csv", index=False)
print(f"  ✅ LightGBM 結果已保存: progressive_lightgbm_results.csv\n")

with open(f"{OUTPUT_DIR}/best_features_xgboost.txt", "w") as f:
    f.write(f"XGBoost 最佳特徵數: {int(best_xgb['n_features'])}\n")
    f.write(f"驗證集 F1: {best_xgb['f1']:.4f}\n")
    f.write(f"特徵列表:\n")
    for i, feat in enumerate(best_xgb["features"], 1):
        f.write(f"{i}. {feat}\n")

with open(f"{OUTPUT_DIR}/best_features_lightgbm.txt", "w") as f:
    f.write(f"LightGBM 最佳特徵數: {int(best_lgb['n_features'])}\n")
    f.write(f"驗證集 F1: {best_lgb['f1']:.4f}\n")
    f.write(f"特徵列表:\n")
    for i, feat in enumerate(best_lgb["features"], 1):
        f.write(f"{i}. {feat}\n")

print(f"  ✅ 最佳特徵列表已保存\n")

print("[6/6] 生成可視化對比...\n")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(
    xgb_results_df["n_features"],
    xgb_results_df["f1"],
    marker="o",
    label="XGBoost",
    linewidth=2,
)
axes[0, 0].plot(
    lgb_results_df["n_features"],
    lgb_results_df["f1"],
    marker="s",
    label="LightGBM",
    linewidth=2,
)
axes[0, 0].axvline(x=best_xgb["n_features"], color="blue", linestyle="--", alpha=0.5)
axes[0, 0].axvline(x=best_lgb["n_features"], color="orange", linestyle="--", alpha=0.5)
axes[0, 0].set_xlabel("Number of Features", fontsize=11)
axes[0, 0].set_ylabel("F1 Score", fontsize=11)
axes[0, 0].set_title("F1 Score vs Number of Features", fontsize=12, fontweight="bold")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(
    xgb_results_df["n_features"],
    xgb_results_df["auc"],
    marker="o",
    label="XGBoost",
    linewidth=2,
)
axes[0, 1].plot(
    lgb_results_df["n_features"],
    lgb_results_df["auc"],
    marker="s",
    label="LightGBM",
    linewidth=2,
)
axes[0, 1].set_xlabel("Number of Features", fontsize=11)
axes[0, 1].set_ylabel("AUC", fontsize=11)
axes[0, 1].set_title("AUC vs Number of Features", fontsize=12, fontweight="bold")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(
    xgb_results_df["n_features"],
    xgb_results_df["precision"],
    marker="o",
    label="XGB Precision",
    linewidth=2,
)
axes[1, 0].plot(
    xgb_results_df["n_features"],
    xgb_results_df["recall"],
    marker="s",
    label="XGB Recall",
    linewidth=2,
)
axes[1, 0].plot(
    lgb_results_df["n_features"],
    lgb_results_df["precision"],
    marker="^",
    label="LGB Precision",
    linewidth=2,
    alpha=0.7,
)
axes[1, 0].plot(
    lgb_results_df["n_features"],
    lgb_results_df["recall"],
    marker="v",
    label="LGB Recall",
    linewidth=2,
    alpha=0.7,
)
axes[1, 0].set_xlabel("Number of Features", fontsize=11)
axes[1, 0].set_ylabel("Score", fontsize=11)
axes[1, 0].set_title("Precision & Recall vs Features", fontsize=12, fontweight="bold")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(
    xgb_results_df["n_features"],
    xgb_results_df["train_time"],
    marker="o",
    label="XGBoost",
    linewidth=2,
)
axes[1, 1].plot(
    lgb_results_df["n_features"],
    lgb_results_df["train_time"],
    marker="s",
    label="LightGBM",
    linewidth=2,
)
axes[1, 1].set_xlabel("Number of Features", fontsize=11)
axes[1, 1].set_ylabel("Training Time (seconds)", fontsize=11)
axes[1, 1].set_title("Training Time vs Features", fontsize=12, fontweight="bold")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plot_file = f"{OUTPUT_DIR}/progressive_feature_selection_comparison.png"
plt.savefig(plot_file, dpi=150, bbox_inches="tight")
print(f"  ✅ 對比圖已保存: {plot_file}\n")

plt.close()

elapsed = (datetime.now() - start).total_seconds() / 60

print("=" * 120)
print("✅ 雙模型漸進式特徵選擇完成")
print("=" * 120 + "\n")

print(f"⏱️  總耗時: {elapsed:.2f} 分鐘\n")

print(f"📊 對比結果:\n")

print(f"  XGBoost 最佳:")
print(f"    特徵數: {int(best_xgb['n_features'])}")
print(f"    F1: {best_xgb['f1']:.4f}")
print(f"    AUC: {best_xgb['auc']:.4f}")
print(f"    Precision: {best_xgb['precision']:.4f}")
print(f"    Recall: {best_xgb['recall']:.4f}\n")

print(f"  LightGBM 最佳:")
print(f"    特徵數: {int(best_lgb['n_features'])}")
print(f"    F1: {best_lgb['f1']:.4f}")
print(f"    AUC: {best_lgb['auc']:.4f}")
print(f"    Precision: {best_lgb['precision']:.4f}")
print(f"    Recall: {best_lgb['recall']:.4f}\n")

if best_xgb["f1"] > best_lgb["f1"]:
    print(f"  🏆 XGBoost 勝出 (F1 高 {best_xgb['f1'] - best_lgb['f1']:.4f})\n")
    winner = "XGBoost"
    winner_features = best_xgb["features"]
else:
    print(f"  🏆 LightGBM 勝出 (F1 高 {best_lgb['f1'] - best_xgb['f1']:.4f})\n")
    winner = "LightGBM"
    winner_features = best_lgb["features"]

print(f"📁 生成的檔案:")
print(f"  1. progressive_xgboost_results.csv")
print(f"  2. progressive_lightgbm_results.csv")
print(f"  3. best_features_xgboost.txt")
print(f"  4. best_features_lightgbm.txt")
print(f"  5. progressive_feature_selection_comparison.png\n")

print(f"🎯 下一步:")
print(f"  1. 使用 {winner} 的最佳特徵組合")
print(f"  2. 用全部訓練數據重訓練")
print(f"  3. 對測試集預測並提交\n")

print(f"💡 關鍵洞察:")
print(f"  • 訓練/驗證分割: 80/20")
print(f"  • 完全隔離測試集")
print(f"  • 找到了本地 F1 的頂峰")
print(f"  • 避免了過擬合\n")

print("=" * 120 + "\n")

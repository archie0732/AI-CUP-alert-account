import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
from path import DATA_PATH, OUTPUT_PATH

warnings.filterwarnings("ignore")

print("=" * 120)
print("🚀 完整優化特徵提取流程（向量化 + 進度條）")
print("=" * 120)

start = datetime.now()

# ==================== [1] Polars 極速載入 ====================
print("[1/6] Polars 極速載入數據...\n")


# 載入交易數據
df_txn = pl.scan_csv(f"{DATA_PATH}/acct_transaction.csv").with_columns(
    [
        pl.col("txn_date").cast(pl.Int32),
        pl.col("txn_amt").cast(pl.Float64, strict=False).fill_null(0.0),
    ]
)

# 載入標籤
df_alert = pl.read_csv(f"{DATA_PATH}/acct_alert.csv")
df_predict = pl.read_csv(f"{DATA_PATH}/acct_predict.csv")

alert_accounts = set(df_alert["acct"].to_list())
test_accounts = set(df_predict["acct"].to_list())

df_txn_collected = df_txn.collect()

print(f"  交易數: {len(df_txn_collected):,}")
print(f"  警示帳戶: {len(alert_accounts):,}")
print(f"  測試帳戶: {len(test_accounts):,}\n")

# ==================== [2] Polars 批量特徵計算 ====================
print("[2/6] Polars 批量計算核心特徵（極速）...\n")

# 核心統計特徵（Polars 向量化）
features_from = df_txn_collected.group_by("from_acct").agg(
    [
        pl.count().alias("out_txn_count"),
        pl.col("txn_amt").sum().alias("total_outflow"),
        pl.col("txn_amt").mean().alias("avg_out_amt"),
        pl.col("txn_amt").std().alias("std_out_amt"),
        pl.col("txn_amt").min().alias("min_out_amt"),
        pl.col("txn_amt").max().alias("max_out_amt"),
        pl.col("txn_date").min().alias("first_out_date"),
        pl.col("txn_date").max().alias("last_out_date"),
        pl.col("txn_date").n_unique().alias("active_out_days"),
        pl.col("to_acct").n_unique().alias("unique_recipients"),
    ]
)

features_to = df_txn_collected.group_by("to_acct").agg(
    [
        pl.count().alias("in_txn_count"),
        pl.col("txn_amt").sum().alias("total_inflow"),
        pl.col("txn_amt").mean().alias("avg_in_amt"),
        pl.col("txn_date").min().alias("first_in_date"),
        pl.col("txn_date").max().alias("last_in_date"),
        pl.col("from_acct").n_unique().alias("unique_senders"),
    ]
)

print(f"  ✅ Polars 核心特徵完成: {(datetime.now() - start).total_seconds():.1f} 秒\n")

# ==================== [3] 合併和衍生特徵 ====================
print("[3/6] 合併特徵並計算衍生特徵...\n")

# 轉為 Pandas
df_features_from = features_from.to_pandas().rename(columns={"from_acct": "account"})
df_features_to = features_to.to_pandas().rename(columns={"to_acct": "account"})

# 合併
df_features = df_features_from.merge(df_features_to, on="account", how="outer").fillna(
    0
)

# 衍生特徵
df_features["total_transactions"] = (
    df_features["out_txn_count"] + df_features["in_txn_count"]
)
df_features["net_flow"] = df_features["total_inflow"] - df_features["total_outflow"]
df_features["turnover_ratio"] = df_features["total_outflow"] / (
    df_features["total_inflow"] + 1
)

# 時間跨度
df_features["first_date"] = df_features[["first_out_date", "first_in_date"]].min(axis=1)
df_features["last_date"] = df_features[["last_out_date", "last_in_date"]].max(axis=1)
df_features["days_span"] = df_features["last_date"] - df_features["first_date"] + 1

# 活躍度
df_features["total_active_days"] = df_features[
    ["active_out_days", "in_txn_count"]
].apply(lambda x: max(x["active_out_days"], 1), axis=1)
df_features["txn_frequency"] = df_features["total_transactions"] / (
    df_features["days_span"] + 1
)

# 對手方多樣性
df_features["total_counterparties"] = (
    df_features["unique_recipients"] + df_features["unique_senders"]
)
df_features["counterparty_concentration"] = df_features["total_transactions"] / (
    df_features["total_counterparties"] + 1
)

print(f"  ✅ 基礎特徵完成\n")

# ==================== [4] 複雜特徵（向量化 + 進度條）====================
print("[4/6] 計算複雜特徵（向量化 + 進度條）...\n")

# 轉為 Pandas
df_txn_pd = df_txn_collected.to_pandas()

# ========== [1/8] 處理時間特徵 ==========
print("  [1/8] 處理時間特徵...")
df_txn_pd["txn_hour"] = pd.to_datetime(
    df_txn_pd["txn_time"], format="%H:%M:%S", errors="coerce"
).dt.hour

# ========== [2/8] 夜間交易比例（向量化）==========
print("  [2/8] 計算夜間交易比例...")
df_txn_pd["is_night"] = (
    (df_txn_pd["txn_hour"] >= 22) | (df_txn_pd["txn_hour"] < 6)
).astype(int)

night_from = df_txn_pd.groupby("from_acct")["is_night"].agg(["sum", "count"])
night_from["night_ratio_from"] = night_from["sum"] / night_from["count"]

night_to = df_txn_pd.groupby("to_acct")["is_night"].agg(["sum", "count"])
night_to["night_ratio_to"] = night_to["sum"] / night_to["count"]

# ========== [3/8] 跨行交易比例（向量化）==========
print("  [3/8] 計算跨行交易比例...")
df_txn_pd["is_other_bank_from"] = (df_txn_pd["to_acct_type"] == "02").astype(int)
df_txn_pd["is_other_bank_to"] = (df_txn_pd["from_acct_type"] == "02").astype(int)

other_bank_from = df_txn_pd.groupby("from_acct")["is_other_bank_from"].agg(
    ["sum", "count"]
)
other_bank_from["other_bank_ratio_from"] = (
    other_bank_from["sum"] / other_bank_from["count"]
)

other_bank_to = df_txn_pd.groupby("to_acct")["is_other_bank_to"].agg(["sum", "count"])
other_bank_to["other_bank_ratio_to"] = other_bank_to["sum"] / other_bank_to["count"]

# ========== [4/8] UNK 通路比例（向量化）==========
print("  [4/8] 計算 UNK 通路比例...")
df_txn_pd["is_unk"] = (df_txn_pd["channel_type"] == "UNK").astype(int)

unk_from = df_txn_pd.groupby("from_acct")["is_unk"].agg(["sum", "count"])
unk_from["unk_ratio_from"] = unk_from["sum"] / unk_from["count"]

unk_to = df_txn_pd.groupby("to_acct")["is_unk"].agg(["sum", "count"])
unk_to["unk_ratio_to"] = unk_to["sum"] / unk_to["count"]

# ========== [5/8] 最後 7 天活躍度（向量化）==========
print("  [5/8] 計算最後 7 天活躍度...")

# 找每個帳戶的最後日期
last_dates_from = df_txn_pd.groupby("from_acct")["txn_date"].max().to_dict()
last_dates_to = df_txn_pd.groupby("to_acct")["txn_date"].max().to_dict()

# 標記最後 7 天的交易
df_txn_pd["is_late_from"] = df_txn_pd.apply(
    lambda x: 1 if x["txn_date"] > last_dates_from.get(x["from_acct"], 0) - 7 else 0,
    axis=1,
)
df_txn_pd["is_late_to"] = df_txn_pd.apply(
    lambda x: 1 if x["txn_date"] > last_dates_to.get(x["to_acct"], 0) - 7 else 0, axis=1
)

late_from = df_txn_pd.groupby("from_acct")["is_late_from"].agg(["sum", "count"])
late_from["late_ratio_from"] = late_from["sum"] / late_from["count"]

late_to = df_txn_pd.groupby("to_acct")["is_late_to"].agg(["sum", "count"])
late_to["late_ratio_to"] = late_to["sum"] / late_to["count"]

# ========== [6/8] 快速轉帳比例（簡化向量化）==========
print("  [6/8] 計算快速轉帳比例...")

quick_from = (
    df_txn_pd.groupby(["from_acct", "txn_date"]).size().reset_index(name="daily_out")
)
quick_to = (
    df_txn_pd.groupby(["to_acct", "txn_date"]).size().reset_index(name="daily_in")
)

quick_merged = quick_from.merge(
    quick_to,
    left_on=["from_acct", "txn_date"],
    right_on=["to_acct", "txn_date"],
    how="inner",
)

quick_count = (
    quick_merged.groupby("from_acct")
    .apply(lambda x: (x[["daily_out", "daily_in"]].min(axis=1)).sum())
    .reset_index(name="quick_count")
)

in_count = df_txn_pd.groupby("to_acct").size().reset_index(name="in_count")
quick_ratio = quick_count.merge(
    in_count, left_on="from_acct", right_on="to_acct", how="left"
).fillna(0)
quick_ratio["quick_transfer_ratio"] = quick_ratio["quick_count"] / (
    quick_ratio["in_count"] + 1
)

# ========== [7/8] 時間熵（批量計算）==========
print("  [7/8] 計算時間熵...")

# 按帳戶收集小時數據
from_hours_dict = (
    df_txn_pd.groupby("from_acct")["txn_hour"]
    .apply(lambda x: x.dropna().values.astype(np.float64))
    .to_dict()
)

to_hours_dict = (
    df_txn_pd.groupby("to_acct")["txn_hour"]
    .apply(lambda x: x.dropna().values.astype(np.float64))
    .to_dict()
)

# 計算時間熵
print("    計算 from_acct 時間熵...")
time_entropy_from_data = []
for acct, hours in tqdm(from_hours_dict.items(), desc="    from_acct", leave=False):
    if len(hours) > 0:
        # 統計每個小時的頻率
        hour_counts = np.zeros(24)
        for h in hours:
            if 0 <= h < 24:
                hour_counts[int(h)] += 1

        # 計算熵
        total = np.sum(hour_counts)
        if total > 0:
            probs = hour_counts / total
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    time_entropy_from_data.append({"account": acct, "time_entropy_from": entropy})

time_entropy_from = pd.DataFrame(time_entropy_from_data)

print("    計算 to_acct 時間熵...")
time_entropy_to_data = []
for acct, hours in tqdm(to_hours_dict.items(), desc="    to_acct", leave=False):
    if len(hours) > 0:
        hour_counts = np.zeros(24)
        for h in hours:
            if 0 <= h < 24:
                hour_counts[int(h)] += 1

        total = np.sum(hour_counts)
        if total > 0:
            probs = hour_counts / total
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    time_entropy_to_data.append({"account": acct, "time_entropy_to": entropy})

time_entropy_to = pd.DataFrame(time_entropy_to_data)

# ========== [8/8] 合併所有複雜特徵 ==========
print("  [8/8] 合併所有複雜特徵...")

# 獲取所有帳戶
all_accounts = df_features["account"].unique()

# 初始化複雜特徵 DataFrame
df_complex = pd.DataFrame({"account": all_accounts})

# 合併夜間交易
df_complex = df_complex.merge(
    night_from[["night_ratio_from"]]
    .reset_index()
    .rename(columns={"from_acct": "account"}),
    on="account",
    how="left",
)
df_complex = df_complex.merge(
    night_to[["night_ratio_to"]].reset_index().rename(columns={"to_acct": "account"}),
    on="account",
    how="left",
)
df_complex["night_txn_ratio"] = df_complex[["night_ratio_from", "night_ratio_to"]].mean(
    axis=1
)

# 合併跨行交易
df_complex = df_complex.merge(
    other_bank_from[["other_bank_ratio_from"]]
    .reset_index()
    .rename(columns={"from_acct": "account"}),
    on="account",
    how="left",
)
df_complex = df_complex.merge(
    other_bank_to[["other_bank_ratio_to"]]
    .reset_index()
    .rename(columns={"to_acct": "account"}),
    on="account",
    how="left",
)
df_complex["other_bank_ratio"] = df_complex[
    ["other_bank_ratio_from", "other_bank_ratio_to"]
].mean(axis=1)

# 合併 UNK 通路
df_complex = df_complex.merge(
    unk_from[["unk_ratio_from"]].reset_index().rename(columns={"from_acct": "account"}),
    on="account",
    how="left",
)
df_complex = df_complex.merge(
    unk_to[["unk_ratio_to"]].reset_index().rename(columns={"to_acct": "account"}),
    on="account",
    how="left",
)
df_complex["unk_channel_ratio"] = df_complex[["unk_ratio_from", "unk_ratio_to"]].mean(
    axis=1
)

# 合併最後 7 天
df_complex = df_complex.merge(
    late_from[["late_ratio_from"]]
    .reset_index()
    .rename(columns={"from_acct": "account"}),
    on="account",
    how="left",
)
df_complex = df_complex.merge(
    late_to[["late_ratio_to"]].reset_index().rename(columns={"to_acct": "account"}),
    on="account",
    how="left",
)
df_complex["late_txn_ratio"] = df_complex[["late_ratio_from", "late_ratio_to"]].mean(
    axis=1
)

# 合併快速轉帳
df_complex = df_complex.merge(
    quick_ratio[["from_acct", "quick_transfer_ratio"]].rename(
        columns={"from_acct": "account"}
    ),
    on="account",
    how="left",
)

# 合併時間熵
df_complex = df_complex.merge(time_entropy_from, on="account", how="left")
df_complex = df_complex.merge(time_entropy_to, on="account", how="left")
df_complex["time_entropy"] = df_complex[["time_entropy_from", "time_entropy_to"]].mean(
    axis=1
)

# 填充缺失值
df_complex = df_complex.fillna(0)

# 計算組合特徵
df_complex = df_complex.merge(
    df_features[["account", "total_active_days", "total_transactions"]],
    on="account",
    how="left",
)

df_complex["cross_bank_intensity"] = df_complex["other_bank_ratio"] * (
    df_complex["total_transactions"] / 100
)
df_complex["late_activity_intensity"] = (
    df_complex["late_txn_ratio"] * df_complex["total_transactions"]
) / (df_complex["total_active_days"] + 1)

# 日均流量
inflow_daily = df_txn_pd.groupby("to_acct")["txn_amt"].sum().reset_index()
inflow_daily = inflow_daily.merge(
    df_features[["account", "total_active_days"]],
    left_on="to_acct",
    right_on="account",
    how="left",
)
inflow_daily["avg_daily_inflow"] = inflow_daily["txn_amt"] / (
    inflow_daily["total_active_days"] + 1
)
inflow_daily = inflow_daily[["to_acct", "avg_daily_inflow"]].rename(
    columns={"to_acct": "account"}
)

outflow_daily = df_txn_pd.groupby("from_acct")["txn_amt"].sum().reset_index()
outflow_daily = outflow_daily.merge(
    df_features[["account", "total_active_days"]],
    left_on="from_acct",
    right_on="account",
    how="left",
)
outflow_daily["avg_daily_outflow"] = outflow_daily["txn_amt"] / (
    outflow_daily["total_active_days"] + 1
)
outflow_daily = outflow_daily[["from_acct", "avg_daily_outflow"]].rename(
    columns={"from_acct": "account"}
)

df_complex = df_complex.merge(inflow_daily, on="account", how="left")
df_complex = df_complex.merge(outflow_daily, on="account", how="left")

df_complex = df_complex.fillna(0)

# 最終合併到主 DataFrame
df_features = df_features.merge(
    df_complex[
        [
            "account",
            "night_txn_ratio",
            "other_bank_ratio",
            "unk_channel_ratio",
            "late_txn_ratio",
            "quick_transfer_ratio",
            "time_entropy",
            "cross_bank_intensity",
            "late_activity_intensity",
            "avg_daily_inflow",
            "avg_daily_outflow",
        ]
    ],
    on="account",
    how="left",
).fillna(0)

print(f"  ✅ 複雜特徵完成: {(datetime.now() - start).total_seconds():.1f} 秒\n")

# ==================== [5] 選擇最終 31 個特徵 ====================
print("[5/6] 選擇最終 31 個特徵（歷史最佳組合）...\n")

# 31 個特徵（基於歷史最佳）
final_31_features = [
    # 基礎統計（10 個）
    "total_transactions",
    "total_inflow",
    "total_outflow",
    "avg_out_amt",
    "avg_in_amt",
    "std_out_amt",
    "min_out_amt",
    "max_out_amt",
    "net_flow",
    "turnover_ratio",
    # 時間特徵（6 個）
    "days_span",
    "total_active_days",
    "txn_frequency",
    "first_date",
    "last_date",
    "late_txn_ratio",
    # 對手方特徵（4 個）
    "unique_recipients",
    "unique_senders",
    "total_counterparties",
    "counterparty_concentration",
    # 行為特徵（7 個）
    "night_txn_ratio",
    "time_entropy",
    "other_bank_ratio",
    "unk_channel_ratio",
    "quick_transfer_ratio",
    "cross_bank_intensity",
    "late_activity_intensity",
    # 日均特徵（4 個）
    "avg_daily_inflow",
    "avg_daily_outflow",
    "out_txn_count",
    "in_txn_count",
]

df_features_final = df_features[["account"] + final_31_features].copy()

print(f"  最終特徵數: {len(final_31_features)}\n")

# ==================== [6] 添加標籤並保存 ====================
print("[6/6] 添加標籤並保存...\n")

# 添加標籤
df_features_final["is_alert"] = (
    df_features_final["account"].isin(alert_accounts).astype(int)
)
df_features_final["is_test"] = (
    df_features_final["account"].isin(test_accounts).astype(int)
)

# 分割訓練集和測試集（修復：使用正確的布爾索引）
train_mask = df_features_final["is_test"] == 0
test_mask = df_features_final["is_test"] == 1

df_train = df_features_final[train_mask].copy()
df_test = df_features_final[test_mask].copy()

# 訓練集標籤
df_train["label"] = df_train["is_alert"]

print(f"  訓練集:")
print(f"    總樣本: {len(df_train):,}")
print(
    f"    警示帳戶: {df_train['label'].sum():,} ({df_train['label'].sum() / len(df_train) * 100:.4f}%)"
)
print(f"    正常帳戶: {(df_train['label'] == 0).sum():,}\n")

print(f"  測試集:")
print(f"    總樣本: {len(df_test):,}\n")

# 保存
train_file = f"{OUTPUT_PATH}/features_31_train_optimized.csv"
test_file = f"{OUTPUT_PATH}/features_31_test_optimized.csv"

df_train[["account", "label"] + final_31_features].to_csv(train_file, index=False)
df_test[["account"] + final_31_features].to_csv(test_file, index=False)

print(f"  ✅ 訓練集已保存: features_31_train_optimized.csv")
print(f"  ✅ 測試集已保存: features_31_test_optimized.csv\n")

# ==================== 最終報告 ====================
elapsed = (datetime.now() - start).total_seconds() / 60

print("=" * 120)
print("✅ 特徵提取完成")
print("=" * 120 + "\n")

print(f"⏱️  總耗時: {elapsed:.2f} 分鐘\n")

print(f"📊 特徵統計:")
print(f"  特徵數: 31 個（歷史最佳）")
print(f"  總帳戶數: {len(df_features_final):,}")
print(f"  訓練集: {len(df_train):,}")
print(f"  測試集: {len(df_test):,}\n")

print(f"✅ 關鍵改進:")
print(f"  1. 提前計算所有帳戶特徵（一次性）")
print(f"  2. Polars 向量化加速")
print(f"  3. 進度條實時顯示")
print(f"  4. 正確分割訓練/測試集（避免泄漏）")
print(f"  5. 31 個歷史最佳特徵\n")

print(f"🎯 下一步:")
print(f"  1. 用這 31 個特徵做漸進式特徵選擇")
print(f"  2. FLAML 超參數優化")
print(f"  3. 正確的訓練/驗證分割")
print(f"  4. 避免數據泄漏\n")

print("=" * 120 + "\n")

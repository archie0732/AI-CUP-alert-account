import pandas as pd
import numpy as np


def calculate_features_optimized(
    account_id, txn_from_grouped, txn_to_grouped, cutoff_date=None
):
    """
    計算帳戶的優化特徵。
    此版本移除了低效率的「快速轉帳」巢狀迴圈，改為更高效的 Pandas 操作。
    """

    try:
        txn_from = txn_from_grouped.get_group(account_id).copy()
    except KeyError:
        txn_from = pd.DataFrame()

    try:
        txn_to = txn_to_grouped.get_group(account_id).copy()
    except KeyError:
        txn_to = pd.DataFrame()

    # 如果有截止日期，只看截止日期前的交易
    if cutoff_date is not None:
        if len(txn_from) > 0:
            txn_from = txn_from[txn_from["txn_date"] < cutoff_date]
        if len(txn_to) > 0:
            txn_to = txn_to[txn_to["txn_date"] < cutoff_date]

    txn_all = pd.concat([txn_from, txn_to])

    if len(txn_all) == 0:
        return None

    # 預先處理基礎數值
    total_txns = len(txn_all)

    features = {}

    # ========== 基礎統計特徵 ==========
    features["total_transactions"] = total_txns
    features["total_inflow"] = txn_to["txn_amt"].sum() if len(txn_to) > 0 else 0.0
    features["total_outflow"] = txn_from["txn_amt"].sum() if len(txn_from) > 0 else 0.0

    features["avg_transaction_amt"] = txn_all["txn_amt"].mean()
    features["min_transaction_amt"] = txn_all["txn_amt"].min()
    features["max_transaction_amt"] = txn_all["txn_amt"].max()
    features["std_transaction_amt"] = (
        txn_all["txn_amt"].std() if total_txns > 1 else 0.0
    )

    # ========== 時間特徵 ==========
    features["active_days"] = txn_all["txn_date"].nunique()
    features["first_txn_date"] = txn_all["txn_date"].min()
    features["last_txn_date"] = txn_all["txn_date"].max()
    features["days_span"] = features["last_txn_date"] - features["first_txn_date"] + 1

    # 最後 7 天活躍度 (使用布林索引優化)
    last_date = txn_all["txn_date"].max()
    late_txns_count = len(txn_all[txn_all["txn_date"] > last_date - 7])
    features["late_transaction_count"] = late_txns_count
    features["late_transaction_ratio"] = (
        late_txns_count / total_txns if total_txns > 0 else 0
    )

    # ========== 資金流特徵 ==========
    features["net_flow"] = features["total_inflow"] - features["total_outflow"]
    features["turnover_ratio"] = features["total_outflow"] / (
        features["total_inflow"] + 1e-6
    )  # 避免除以零

    # ========== 交易對手特徵 ==========
    counterparties = []
    if len(txn_from) > 0:
        counterparties.extend(txn_from["to_acct"].values)
    if len(txn_to) > 0:
        counterparties.extend(txn_to["from_acct"].values)

    # 過濾掉自己
    counterparties = [c for c in counterparties if c != account_id]
    unique_counterparties = len(set(counterparties))
    features["unique_counterparty_count"] = unique_counterparties
    features["counterparty_concentration"] = total_txns / (unique_counterparties + 1)

    # ========== 跨行交易特徵 ==========
    other_bank_count = 0
    if len(txn_from) > 0:
        other_bank_count += len(txn_from[txn_from["to_acct_type"] == 2])
    if len(txn_to) > 0:
        other_bank_count += len(txn_to[txn_to["from_acct_type"] == 2])

    features["other_bank_count"] = other_bank_count
    features["other_bank_ratio"] = other_bank_count / total_txns

    # ========== 時間模式特徵 ==========
    # 夜間交易
    night_txns_count = len(
        txn_all[(txn_all["txn_hour"] >= 22) | (txn_all["txn_hour"] < 6)]
    )
    features["night_transaction_count"] = night_txns_count
    features["night_transaction_ratio"] = night_txns_count / total_txns

    # 時間熵
    hour_counts = txn_all["txn_hour"].value_counts(normalize=True)
    features["time_entropy"] = -np.sum(hour_counts * np.log2(hour_counts + 1e-10))

    # ========== 通路特徵 ==========
    features["channel_diversity"] = txn_all["channel_type"].nunique()
    unk_channel_count = len(txn_all[txn_all["channel_type"] == "UNK"])
    features["unk_channel_ratio"] = unk_channel_count / total_txns

    # ========== 二階特徵（組合特徵）==========
    features["cross_bank_intensity"] = (
        features["other_bank_ratio"] * features["unique_counterparty_count"]
    )
    features["late_activity_intensity"] = features["late_transaction_count"] / (
        features["active_days"] + 1e-6
    )
    features["txn_frequency"] = total_txns / (features["days_span"] + 1e-6)
    features["avg_daily_inflow"] = features["total_inflow"] / (
        features["active_days"] + 1e-6
    )
    features["avg_daily_outflow"] = features["total_outflow"] / (
        features["active_days"] + 1e-6
    )

    # 新增時間序列特徵 (替代耗時的 quick_transfer)
    # 每次交易的金額標準差 (反映交易行為的規律性)
    features["txn_amt_std_by_day"] = (
        txn_all.groupby("txn_date")["txn_amt"].std().mean()
        if features["active_days"] > 1
        else 0.0
    )

    return features


def generate_all_features(
    accounts,
    txn_from_grouped,
    txn_to_grouped,
    alert_event_dict=None,
    is_alert_set=False,
):
    """為一組帳戶生成所有特徵並打包成 DataFrame。"""
    all_features = []
    total_accounts = len(accounts)

    for idx, acct in enumerate(accounts):
        if (idx + 1) % 500 == 0:
            set_type = "警示帳戶" if is_alert_set else "測試帳戶"
            print(f"  ---> 進度: {set_type} {idx+1}/{total_accounts}")

        cutoff_date = alert_event_dict.get(acct) if is_alert_set else None

        # 使用優化後的特徵計算函數
        features = calculate_features_optimized(
            acct, txn_from_grouped, txn_to_grouped, cutoff_date=cutoff_date
        )

        if features:
            features["acct"] = acct
            all_features.append(features)

    return pd.DataFrame(all_features)

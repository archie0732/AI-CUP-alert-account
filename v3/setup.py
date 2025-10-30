import pandas as pd
from datetime import datetime


def setup_data(DATA_PATH: str):
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 步驟 1/4: 載入原始數據與建立索引..."
    )

    try:
        df_txn = pd.read_csv(f"{DATA_PATH}/acct_transaction.csv")
        df_alert = pd.read_csv(f"{DATA_PATH}/acct_alert.csv")
        df_predict = pd.read_csv(f"{DATA_PATH}/acct_predict.csv")
    except FileNotFoundError as e:
        print(f"⚠️ 錯誤：找不到文件。請確認 DATA_PATH ({DATA_PATH}) 是否正確。")
        raise e

    df_txn["txn_datetime"] = pd.to_datetime(
        df_txn["txn_date"].astype(str) + " " + df_txn["txn_time"].astype(str),
        format="%d %H:%M:%S",
        errors="coerce",
    )
    df_txn["txn_hour"] = df_txn["txn_datetime"].dt.hour

    df_txn.dropna(subset=["txn_datetime"], inplace=True)

    alert_accounts = set(df_alert["acct"].unique())
    test_accounts = set(df_predict["acct"].unique())
    alert_event_dict = dict(zip(df_alert["acct"], df_alert["event_date"]))

    print(f"  交易記錄：{len(df_txn):,} 筆")
    print(f"  警示帳戶：{len(alert_accounts)} 個")
    print(f"  測試帳戶：{len(test_accounts)} 個")

    txn_from_grouped = df_txn.groupby("from_acct")
    txn_to_grouped = df_txn.groupby("to_acct")

    return (
        df_txn,
        df_alert,
        df_predict,
        txn_from_grouped,
        txn_to_grouped,
        alert_event_dict,
        alert_accounts,
        test_accounts,
    )

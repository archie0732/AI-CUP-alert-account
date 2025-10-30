import pandas as pd
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from setup import setup_data
from utils import generate_all_features
from path import OUTPUT_PATH, CONTAMINATION_VALUES, N_ESTIMATORS, RANDOM_STATE


def main():
    print("=" * 80)
    print("🎯 One-Class Learning: Isolation Forest (優化版)")
    print("=" * 80)

    start_time = datetime.now()
    print(f"開始時間: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 數據準備
    (
        df_txn,
        _,
        df_predict,
        txn_from_grouped,
        txn_to_grouped,
        alert_event_dict,
        alert_accounts,
        test_accounts,
    ) = setup_data()

    # 2. 特徵生成
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 步驟 2/4: 生成特徵 (已優化)...")

    # 警示帳戶特徵
    df_alert_features = generate_all_features(
        alert_accounts,
        txn_from_grouped,
        txn_to_grouped,
        alert_event_dict,
        is_alert_set=True,
    )
    print(f"  ✅ 警示帳戶特徵: {len(df_alert_features)} 個")

    # 測試集特徵
    df_test_features = generate_all_features(
        test_accounts, txn_from_grouped, txn_to_grouped
    )
    print(f"  ✅ 測試集特徵: {len(df_test_features)} 個")

    # 3. 訓練與預測
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] 步驟 3/4: 訓練 Isolation Forest..."
    )

    feature_cols = [col for col in df_alert_features.columns if col not in ["acct"]]

    X_alert = df_alert_features[feature_cols].fillna(0)
    X_test = df_test_features[feature_cols].fillna(0)

    print(f"  訓練數據: {len(X_alert)} 個警示帳戶")
    print(f"  測試數據: {len(X_test)} 個帳戶")
    print(f"  特徵數量: {len(feature_cols)} 個")

    # 標準化
    print(f"\n  標準化特徵...")
    scaler = StandardScaler()
    X_alert_scaled = scaler.fit_transform(X_alert)
    X_test_scaled = scaler.transform(X_test)

    # 訓練多個模型
    print(f"\n  訓練 {len(CONTAMINATION_VALUES)} 個模型...")
    results = {}

    for contamination in CONTAMINATION_VALUES:
        print(f"  訓練 contamination={contamination:.2f}...", end=" ")

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(X_alert_scaled)

        y_pred = model.predict(X_test_scaled)
        # -1 = 異常 (標註為 1), 1 = 正常 (標註為 0)
        y_pred_binary = (y_pred == -1).astype(int)

        pred_count = y_pred_binary.sum()
        pred_ratio = pred_count / len(df_test_features) * 100

        results[contamination] = {
            "predictions": y_pred_binary,
            "count": pred_count,
            "ratio": pred_ratio,
        }

        print(f"預測 {pred_count:4d} 個 ({pred_ratio:5.2f}%)")

    # 4. 生成提交文件
    print(
        f"\n[{datetime.now().strftime('%H:%M:%S')}] 步驟 4/4: 生成提交文件與保存特徵..."
    )

    # 保存特徵
    df_alert_features.to_csv(
        f"{OUTPUT_PATH}/oneclass_alert_features_optimized.csv", index=False
    )
    df_test_features.to_csv(
        f"{OUTPUT_PATH}/oneclass_test_features_optimized.csv", index=False
    )

    print(f"  💾 特徵已保存:")
    print(f"    - oneclass_alert_features_optimized.csv")
    print(f"    - oneclass_test_features_optimized.csv")

    for contamination, result in results.items():
        df_submission = pd.DataFrame(
            {"acct": df_test_features["acct"], "label": result["predictions"]}
        )

        filename = f"{OUTPUT_PATH}/submission_oneclass_if_{int(contamination*100):02d}_optimized.csv"
        df_submission.to_csv(filename, index=False)

        print(
            f"  ✅ contamination {contamination:.2f}: {result['count']:4d} 個 → {filename}"
        )

    # 總結
    total_time = (datetime.now() - start_time).total_seconds() / 60

    print(f"\n{'='*80}")
    print(f"🎉 One-Class Learning (優化版) 完成！")
    print(f"{'='*80}")
    print(f"⏱️  總運行時間: {total_time:.1f} 分鐘")
    print(
        f"📊 訓練數據: {len(df_alert_features)} 個警示帳戶 | 特徵數量: {len(feature_cols)} 個"
    )
    print(f"📁 生成文件: {len(CONTAMINATION_VALUES)} 個提交文件 (帶有 _optimized 標記)")

    print(f"\n📊 預測數量統計:")
    print(f"  contamination   預測數量   比例")
    print(f"  " + "-" * 40)
    for contamination in CONTAMINATION_VALUES:
        result = results[contamination]
        print(
            f"  {contamination:.2f}             {result['count']:4d}     {result['ratio']:5.2f}%"
        )

    print(f"\n{'='*80}")
    print(f"✅ 所有文件已準備好！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

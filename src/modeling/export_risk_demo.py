#!/usr/bin/env python3
"""Export full-channel retrospective risk cards from the frozen model."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from improve_baseline import augment, frame, ROOT, OUT
from predict_risk import score_features


def main():
    start = time.time()
    mart = augment(pd.read_parquet(ROOT / "data/interim/baseline_mart_2025_2026.parquet"))
    day = mart.date.max()
    latest = mart[mart.date == day].copy()
    selection = json.loads((OUT / "selection.json").read_text())
    latest["score"] = score_features(latest, OUT).score.to_numpy()
    # Do not portray an unobserved channel as a healthy operating device.
    eligible = latest[latest.observed_days_7d > 0].sort_values("score", ascending=False)
    cards = eligible.head(100)
    channels = pd.read_csv(ROOT / "data/raw/справочник_каналов_датчиков.csv", dtype=str).set_index("ид_канала_данных")
    objects = pd.read_csv(ROOT / "data/raw/справочник_объектов_диспетчер.csv", dtype=str).set_index("ид_объект")
    explanations = {}
    if selection["selected_model"].startswith("catboost"):
        model = CatBoostClassifier()
        model.load_model(str(OUT / "models" / f"{selection['selected_model']}.cbm"))
        x = frame(cards, cards.index, selection["features"], selection["categorical_features"], True)
        values = model.get_feature_importance(Pool(x, cat_features=selection["categorical_features"]), type="ShapValues", thread_count=6)
        for pos, index in enumerate(cards.index):
            weights = values[pos, :-1]
            top = [int(j) for j in np.argsort(weights)[::-1] if weights[j] > 0][:3]
            explanations[index] = [{"feature": selection["features"][j], "value": str(x.iloc[pos, j]),
                                    "contribution_log_odds": float(weights[j])} for j in top]
    items = []
    for index, row in cards.iterrows():
        cid = str(row.channel_id)
        meta = channels.loc[cid] if cid in channels.index else {}
        oid = str(meta.get("ид_объект", "unknown"))
        obj = objects.loc[oid] if oid in objects.index else {}
        items.append({"id": f"{day.date()}_{cid}", "channel_id": cid, "object_id": oid,
            "channel_name": str(meta.get("название_датчика", cid)),
            "sensor_type": str(meta.get("тип_датчика", "Нет в справочнике")),
            "object_name": str(obj.get("диспетчерское_название_объекта", f"Объект {oid}")),
            "score": float(row.score), "warning": bool(row.score >= selection["threshold"]),
            "current_alarm": bool(row.alarm_count > 0), "events_today": int(row.events_count),
            "alarm_days_7d": int(row.alarm_days_7d), "observed_days_7d": int(row.observed_days_7d),
            "days_since_alarm": int(row.days_since_alarm), "explanation": explanations.get(index, []),
            "suggested_action": "Проверить журнал и работоспособность канала; подтвердить необходимость обслуживания.",
            "actual_next_day_alarm": bool(row.target_alarm_next_24h)})
    metrics = pd.read_csv(OUT / "comparison.csv")
    selected = metrics[(metrics.split == "test") & (metrics.model != "archived_baseline")].iloc[0]
    export = {"mode": "retrospective", "feature_date": str(day.date()),
        "forecast_start": (day + pd.Timedelta(days=1)).isoformat(),
        "forecast_end": (day + pd.Timedelta(days=2)).isoformat(),
        "model": selection["selected_model"], "threshold": selection["threshold"],
        "total_channels": len(latest), "observed_channels_7d": len(eligible), "shown_cards": len(items),
        "warnings_observed_7d": int((eligible.score >= selection["threshold"]).sum()),
        "test_metrics": {k:float(selected[k]) for k in ["average_precision", "precision", "recall", "f1"]},
        "limitations": "Прогноз сообщения о тревоге, не подтверждённого отказа. Score не калиброван. Архивная демонстрация.",
        "cards": items}
    destination = ROOT / "data/app/risk_demo.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(export, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Exported {len(items)} cards from {len(latest)} channels in {time.time()-start:.1f}s: {destination}")


if __name__ == "__main__":
    main()

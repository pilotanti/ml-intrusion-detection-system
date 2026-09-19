"""Train the ML intrusion detection model on NSL-KDD.

Trains a Random Forest that classifies each network connection as Normal or
one of four attack families (DoS, Probe, R2L, U2R). It is fitted on
KDDTrain+ and evaluated on the separate KDDTest+ file, which contains attack
types never seen in training, so the reported test scores are realistic.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ids_common import (CATEGORICAL, FEATURES, MODEL_FILENAME, NUMERIC,
                        load_records, to_family)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SAMPLE_DIR = ROOT / "samples"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    train = load_records(DATA_DIR / "KDDTrain+.txt")
    test = load_records(DATA_DIR / "KDDTest+.txt")
    y_train, y_test = to_family(train["label"]), to_family(test["label"])
    print(f"Train: {len(train):,} rows   Test: {len(test):,} rows")
    print("Train class mix:", y_train.value_counts().to_dict())

    model = build_pipeline()
    t0 = time.time()
    model.fit(train[FEATURES], y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    pred = model.predict(test[FEATURES])
    is_attack_true = y_test != "Normal"
    is_attack_pred = pred != "Normal"

    metrics = {
        "multiclass_accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "binary_accuracy": accuracy_score(is_attack_true, is_attack_pred),
        "attack_precision": precision_score(is_attack_true, is_attack_pred),
        "attack_recall": recall_score(is_attack_true, is_attack_pred),
        "false_alarm_rate": float(np.mean(is_attack_pred[~is_attack_true.values])),
    }
    labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    report = classification_report(y_test, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)

    print("\n=== Evaluation on KDDTest+ ===")
    for k, v in metrics.items():
        print(f"{k:>22}: {v:.4f}")
    print("\n" + report)
    print("Confusion matrix (rows = true, cols = predicted):", labels)
    print(cm)

    # Attack is flagged when P(attack) = 1 - P(Normal) >= threshold. Lower
    # thresholds catch more (unseen) attacks at the cost of more false alarms.
    p_attack = 1 - model.predict_proba(test[FEATURES])[:, list(model.classes_).index("Normal")]
    sweep = []
    print("\nThreshold sweep (binary attack detection):")
    for th in (0.5, 0.4, 0.3, 0.2, 0.1):
        flag = p_attack >= th
        tp = int((flag & is_attack_true.values).sum())
        fp = int((flag & ~is_attack_true.values).sum())
        row = {"threshold": th,
               "accuracy": float((flag == is_attack_true.values).mean()),
               "recall": tp / int(is_attack_true.sum()),
               "precision": tp / max(tp + fp, 1),
               "false_alarm_rate": fp / int((~is_attack_true).sum())}
        sweep.append(row)
        print("  th={threshold:.2f} acc={accuracy:.3f} recall={recall:.3f} "
              "precision={precision:.3f} FAR={false_alarm_rate:.3f}".format(**row))

    importances = model.named_steps["clf"].feature_importances_
    names = model.named_steps["pre"].get_feature_names_out()
    top = sorted(zip(names, importances), key=lambda t: -t[1])[:10]

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "classes": list(model.classes_),
        "features": FEATURES,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_labels": labels,
        "top_features": [(n.split("__", 1)[-1], float(v)) for n, v in top],
        "threshold_sweep": sweep,
        "default_threshold": 0.3,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": len(train),
        "test_rows": len(test),
        "dataset": "NSL-KDD (KDDTrain+ / KDDTest+)",
    }
    out = MODEL_DIR / MODEL_FILENAME
    joblib.dump(bundle, out, compress=3)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(
        {"argmax": {k: round(v, 4) for k, v in metrics.items()},
         "threshold_sweep": [{k: round(v, 4) for k, v in r.items()} for r in sweep]},
        indent=2))
    print(f"\nModel saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Small labelled sample of unseen test traffic for demos inside the app.
    SAMPLE_DIR.mkdir(exist_ok=True)
    sample = test.sample(n=2000, random_state=7)[FEATURES + ["label"]]
    sample.to_csv(SAMPLE_DIR / "sample_traffic.csv", index=False)
    print(f"Sample traffic written to {SAMPLE_DIR / 'sample_traffic.csv'}")


if __name__ == "__main__":
    main()

"""Đo chất lượng và tổng hợp các lần chạy thành báo cáo so sánh."""

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

from data_utils import LABEL_NAMES, LABELS, ROOT, configure_console

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCORES = ["macro_f1", "weighted_f1", "accuracy", "f1_0", "f1_1", "f1_2"]
MEASURES = SCORES + ["training_seconds", "inference_ms_per_1000", "model_size_mib"]
GROUPS = ["scenario", "held_out", "scope", "model", "variant"]


def macro_f1(labels, predictions):
    return f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def save_preprocessor(path, processor):
    config = asdict(processor)
    config.pop("_entity_pattern")
    write_json(path, config)


def run_directory(args, model_name, split, seed):
    model_slug = re.sub(r'[<>:"/\\|?*]+', "--", model_name).strip(". ")
    if not model_slug or not args.variant or not all(
        character.isalnum() or character in "-_" for character in args.variant
    ):
        raise ValueError("Tên model/variant không dùng được làm tên thư mục.")
    directory = (args.output_dir / "runs" / model_slug / args.variant / split.scenario
                 / split.held_out / f"fold-{split.fold}" / f"seed-{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").unlink(missing_ok=True)
    return directory


def split_signature(split):
    digest = hashlib.sha256()
    for partition in (split.train, split.val, split.test):
        rows = partition[["text", "label", "domain"]].to_csv(index=False, lineterminator="\n")
        digest.update(rows.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def benchmark(infer, frame, split_seed):
    # Cùng một mẫu 1.000 dòng cho mọi mô hình trên một tập test.
    sample = frame.sample(n=1000, replace=len(frame) < 1000, random_state=split_seed)
    texts = sample["text"].tolist()
    infer(texts)
    elapsed = []
    for _ in range(3):
        start = perf_counter()
        infer(texts)
        elapsed.append((perf_counter() - start) * 1000)
    return median(elapsed)


def save_evaluation(directory, split, probabilities, model, seed, args, *,
                    training_seconds, inference_ms, model_path, device):
    predictions = split.test[["text", "domain", "label"]].rename(columns={"label": "y_true"}).copy()
    predictions["y_pred"] = probabilities.argmax(axis=1)
    for label in LABELS:
        predictions[f"prob_{label}"] = probabilities[:, label]
    predictions.to_csv(directory / "predictions.csv", index=False, encoding="utf-8-sig")
    errors = predictions.loc[predictions["y_true"].ne(predictions["y_pred"])].copy()
    errors["negative_error"] = errors["y_true"].eq(1) | errors["y_pred"].eq(1)
    errors["confidence"] = errors[[f"prob_{label}" for label in LABELS]].max(axis=1)
    errors.sort_values(["negative_error", "confidence"], ascending=False).to_csv(
        directory / "errors.csv", index=False, encoding="utf-8-sig"
    )

    size = (model_path.stat().st_size if model_path.is_file()
            else sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file()))
    metadata = {"model": model, "variant": args.variant, "seed": seed, "fold": split.fold,
                "scenario": split.scenario, "held_out": split.held_out,
                "split_signature": split_signature(split), "split_seed": args.split_seed,
                "training_seconds": training_seconds, "inference_ms_per_1000": inference_ms,
                "model_size_mib": size / 1024**2, "device": device,
                "n_train": len(split.train), "n_val": len(split.val)}
    records = []
    for scope, frame in [("all", predictions), *predictions.groupby("domain")]:
        y_true, y_pred = frame["y_true"], frame["y_pred"]
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
        figure, axis = plt.subplots(figsize=(5.3, 4.4))
        ConfusionMatrixDisplay(matrix, display_labels=LABEL_NAMES).plot(
            ax=axis, cmap="Blues", colorbar=False, values_format="d"
        )
        axis.set_title(f"{model} · {scope}")
        figure.tight_layout()
        figure.savefig(directory / f"confusion_{scope}.png", dpi=150)
        plt.close(figure)
        record = dict(metadata, scope=scope, n_test=len(frame), accuracy=accuracy_score(y_true, y_pred),
                      macro_f1=float(f1.mean()), confusion_matrix=matrix.tolist(),
                      weighted_f1=f1_score(y_true, y_pred, average="weighted", zero_division=0))
        for label in LABELS:
            record.update({f"precision_{label}": precision[label], f"recall_{label}": recall[label],
                           f"f1_{label}": f1[label], f"support_{label}": int(support[label])})
        records.append(record)
    write_json(directory / "config.json", vars(args))
    write_json(directory / "metrics.json", records)
    return records[0]


def summarize(frame):
    rows = []
    for keys, group in frame.groupby(GROUPS, sort=True):
        # Khi chạy cả nhiều seed lẫn nhiều fold, đo hai nguồn biến thiên riêng.
        per_seed = group.groupby("seed")[MEASURES].mean()
        per_fold = group.groupby("fold")[MEASURES].mean()
        row = dict(zip(GROUPS, keys), runs=len(group), seeds=len(per_seed), folds=len(per_fold),
                   min_class_support=int(group[[f"support_{label}" for label in LABELS]].min().min()),
                   devices=", ".join(sorted(group["device"].unique())))
        for metric in MEASURES:
            row.update({f"{metric}_mean": group[metric].mean(),
                        f"{metric}_seed_std": per_seed[metric].std(),
                        f"{metric}_fold_std": per_fold[metric].std()})
        rows.append(row)
    return pd.DataFrame(rows)


def ablation(frame):
    keys = GROUPS[:-1] + ["seed", "fold", "split_signature"]
    main = frame.loc[frame["variant"].eq("main"), keys + ["macro_f1"]]
    changes = frame.loc[~frame["variant"].eq("main")].merge(
        main, on=keys, suffixes=("", "_main"), validate="many_to_one"
    )
    changes["delta"] = changes["macro_f1"] - changes["macro_f1_main"]
    rows = []
    for values, group in changes.groupby(GROUPS):
        rows.append(dict(zip(GROUPS, values), pairs=len(group),
                         delta_macro_f1=group["delta"].mean(),
                         delta_seed_std=group.groupby("seed")["delta"].mean().std(),
                         delta_fold_std=group.groupby("fold")["delta"].mean().std()))
    return pd.DataFrame(rows, columns=GROUPS + [
        "pairs", "delta_macro_f1", "delta_seed_std", "delta_fold_std"
    ])


def result_notes(frame, summary):
    lines = ["# Nhận xét kết quả", "",
             "Macro-F1 gồm cả ba nhãn. Độ lệch chuẩn mẫu qua seed/fold mô tả độ biến thiên. "
             "Các seed dùng chung tập test và các fold có tập train chồng lấp, nên độ lệch chuẩn "
             "chỉ hỗ trợ nhận định mô tả về chênh lệch.", "",
             "Thời gian suy luận đo trên thiết bị ghi trong bảng, gồm tiền xử lý và dự đoán xác suất.", ""]
    main = frame.loc[frame["variant"].eq("main")]
    by_scope = ["scenario", "held_out", "scope"]
    for keys, group in summary.loc[summary["variant"].eq("main")].groupby(by_scope):
        if keys[0] == "cross-domain" and keys[2] != "all":
            continue
        lines.extend([f"## {' / '.join(keys)}", ""])
        if group["min_class_support"].min() < 20:
            lines.extend(["Một lớp có dưới 20 mẫu test. Cần thêm dữ liệu để kết luận chắc chắn.", ""])
        selected = main.loc[(main[by_scope] == pd.Series(dict(zip(by_scope, keys)))).all(axis=1)]
        baseline = selected.loc[selected["model"].eq("tfidf-lr")]
        for model, candidate in selected.loc[~selected["model"].eq("tfidf-lr")].groupby("model"):
            if baseline.empty:
                continue
            # LR chạy một lần cho mỗi fold. So sánh BERT với đúng các fold chung.
            pairs = candidate.merge(baseline, on=["fold", "split_signature"], suffixes=("", "_lr"))
            if len(pairs) != len(candidate) or set(pairs["fold"]) != set(baseline["fold"]):
                lines.extend([f"- {model} và TF-IDF+LR có tập chia khác nhau hoặc thiếu fold chung.", ""])
                continue
            delta = pairs["macro_f1"] - pairs["macro_f1_lr"]
            class_deltas = [f"{LABEL_NAMES[label]} {((pairs[f'f1_{label}'] - pairs[f'f1_{label}_lr']).mean()):+.4f}"
                            for label in LABELS]
            seed_std = delta.groupby(pairs["seed"]).mean().std()
            fold_std = delta.groupby(pairs["fold"]).mean().std()
            deviations = [value for value in (seed_std, fold_std) if pd.notna(value)]
            relation = ("Mức chênh nằm trong một độ lệch chuẩn quan sát."
                        if deviations and abs(delta.mean()) <= max(deviations)
                        else "Mức chênh lớn hơn độ lệch chuẩn quan sát."
                        if deviations else "Cần thêm seed hoặc fold để đánh giá độ biến thiên.")
            std_seed_text = f"{seed_std:.4f}" if pd.notna(seed_std) else "—"
            std_fold_text = f"{fold_std:.4f}" if pd.notna(fold_std) else "—"
            lines.extend([f"- {model} so với TF-IDF+LR đạt Δ macro-F1 {delta.mean():+.4f} "
                          f"(SD seed {std_seed_text}, SD fold {std_fold_text}). "
                          f"Δ F1 từng lớp gồm {', '.join(class_deltas)}. {relation}", ""])
            if pairs["seed"].nunique() < 3:
                lines.extend([f"  {model} có dưới ba seed trên tập chia này.", ""])
        if selected["model"].nunique() == 1:
            lines.extend(["Bảng hiện có kết quả của một mô hình cho tập chia này.", ""])
    return "\n".join(lines) + "\n"


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/sentiment")
    args = parser.parse_args()
    paths = sorted((args.output_dir / "runs").rglob("metrics.json"))
    if not paths:
        parser.error("Chưa có metrics.json. Hãy chạy ít nhất một chương trình huấn luyện.")
    frame = pd.DataFrame(row for path in paths for row in json.loads(path.read_text(encoding="utf-8")))
    summary = summarize(frame)
    frame.to_csv(args.output_dir / "run_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    ablation(frame).to_csv(args.output_dir / "ablation.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "analysis.md").write_text(result_notes(frame, summary), encoding="utf-8")
    columns = GROUPS + ["runs", "macro_f1_mean", "macro_f1_seed_std", "macro_f1_fold_std"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Đã lưu báo cáo tại {args.output_dir}")


if __name__ == "__main__":
    main()

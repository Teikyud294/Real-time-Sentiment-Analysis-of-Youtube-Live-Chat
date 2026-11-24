"""Huấn luyện baseline TF-IDF mức từ và ký tự với Logistic Regression."""

import argparse
import logging
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from data_utils import (EMOTE, LABEL_NAMES, LABELS, PLACEHOLDER, add_common_args,
                        configure_console, cv_indices, load_data, make_preprocessor, make_splits)
from evaluate import benchmark, macro_f1, run_directory, save_evaluation, save_preprocessor

WORD_TOKEN = rf"(?:{EMOTE}|{PLACEHOLDER}|\b\d+(?:[-:]\d+)+\b|\b\w+\b|[^\s\w])"


def make_pipeline(args):
    features = FeatureUnion([
        ("word", TfidfVectorizer(token_pattern=WORD_TOKEN, ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, lowercase=False, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                                 sublinear_tf=True, lowercase=False, dtype=np.float32)),
    ])
    classifier = LogisticRegression(solver="saga", max_iter=args.max_iter, C=args.c, tol=args.tol,
                                    class_weight="balanced" if args.class_weight else None,
                                    random_state=args.seed)
    return Pipeline([("features", features), ("classifier", classifier)])


def fit_model(args, data, preprocess):
    pipeline = make_pipeline(args)
    start = perf_counter()
    texts = data["text"].map(preprocess)
    if args.search == "none":
        pipeline.fit(texts, data["label"])
        return pipeline, None, perf_counter() - start

    support = data.groupby("label")["key"].nunique().reindex(LABELS, fill_value=0)
    folds = min(args.search_folds, int(support.min()))
    if folds < 2:
        raise ValueError("Cần ít nhất hai nhóm mỗi lớp trong train để tìm siêu tham số.")
    params = {
        "classifier__C": [0.25, 1.0, 4.0],
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "features__char__ngram_range": [(2, 5), (3, 5)],
        "features__word__min_df": [1, 2],
        "features__word__max_features": [30_000, 80_000],
        "features__word__strip_accents": [None, "unicode"],
    }
    options = dict(scoring=make_scorer(macro_f1), cv=cv_indices(data, folds, args.seed),
                   n_jobs=args.n_jobs, refit=True, verbose=args.search_verbose, error_score="raise")
    if args.search == "grid":
        search = GridSearchCV(pipeline, params, **options)
    else:
        search = RandomizedSearchCV(pipeline, params, n_iter=args.search_iterations,
                                    random_state=args.seed, **options)
    search.fit(texts, data["label"])
    logging.info("CV macro-F1=%.4f | tham số tốt nhất %s", search.best_score_, search.best_params_)
    return search.best_estimator_, pd.DataFrame(search.cv_results_), perf_counter() - start


def save_top_features(pipeline, path):
    names = pipeline["features"].get_feature_names_out()
    rows = []
    for label, coefficients in zip(pipeline["classifier"].classes_, pipeline["classifier"].coef_):
        indexes = np.argsort(coefficients)[-15:][::-1]
        logging.info("Top 15 %s: %s", LABEL_NAMES[label], ", ".join(names[indexes]))
        rows.extend({"label": label, "class": LABEL_NAMES[label], "rank": rank,
                     "feature": names[index], "coefficient": coefficients[index]}
                    for rank, index in enumerate(indexes, 1))
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--c", type=float, default=1.0, help="C khi không chạy tìm kiếm")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--search", choices=("random", "grid", "none"), default="random")
    parser.add_argument("--search-folds", type=int, default=5)
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--search-verbose", type=int, default=1)
    args = parser.parse_args()
    if args.search_folds < 2 or min(args.max_iter, args.search_iterations) < 1 or min(args.c, args.tol) <= 0:
        parser.error("Cần search-folds ≥ 2, max-iter và search-iterations ≥ 1, C và tol > 0.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data = load_data(args)
    preprocess = make_preprocessor(args)
    for split in make_splits(data, args):
        logging.info("%s / %s / fold %d | train=%d val=%d test=%d", split.scenario,
                     split.held_out, split.fold, len(split.train), len(split.val), len(split.test))
        directory = run_directory(args, "tfidf-lr", split, args.seed)
        pipeline, search_results, seconds = fit_model(args, split.train, preprocess)
        model_path = directory / "model.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        save_preprocessor(directory / "preprocessing.json", preprocess)
        save_top_features(pipeline, directory / "top_features.csv")
        search_path = directory / "search.csv"
        if search_results is not None:
            search_results.sort_values("rank_test_score").to_csv(search_path, index=False)
        else:
            search_path.unlink(missing_ok=True)

        def infer(texts):
            return pipeline.predict_proba([preprocess(text) for text in texts])

        logging.info("Validation macro-F1=%.4f", macro_f1(split.val["label"],
                     infer(split.val["text"]).argmax(axis=1)))
        metrics = save_evaluation(
            directory, split, infer(split.test["text"]), "tfidf-lr", args.seed, args,
            training_seconds=seconds, inference_ms=benchmark(infer, split.test, args.split_seed),
            model_path=model_path, device="cpu",
        )
        logging.info("Test macro-F1=%.4f | %s", metrics["macro_f1"], directory)


if __name__ == "__main__":
    main()

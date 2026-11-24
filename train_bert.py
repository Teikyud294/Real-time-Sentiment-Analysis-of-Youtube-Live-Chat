"""Fine-tune ViSoBERT, PhoBERT hoặc XLM-R với Hugging Face Trainer."""

import argparse
import inspect
import logging
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments, set_seed)

from data_utils import (LABEL_NAMES, LABELS, add_common_args, configure_console, load_data,
                        make_preprocessor, make_splits)
from evaluate import benchmark, macro_f1, run_directory, save_evaluation, save_preprocessor, write_json


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Loss đã lấy trung bình theo batch. Trainer tự chia khi tích lũy gradient.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{name: value for name, value in inputs.items() if name != "labels"})
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels,
                                                 weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


def dataset_for(frame, tokenizer, preprocess, max_length):
    dataset = Dataset.from_dict({"text": frame["text"].tolist(), "labels": frame["label"].tolist()})

    def encode(batch):
        return tokenizer([preprocess(text) for text in batch["text"]], truncation=True,
                         max_length=max_length)

    return dataset.map(encode, batched=True, remove_columns=["text"])


def training_args(args, scratch, seed, lr, fp16):
    # Transformers 4 dùng warmup_ratio. Transformers 5 nhận tỷ lệ ở warmup_steps.
    parameters = inspect.signature(TrainingArguments).parameters
    warmup = "warmup_ratio" if "warmup_ratio" in parameters else "warmup_steps"
    return TrainingArguments(
        output_dir=str(scratch), learning_rate=lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay, fp16=fp16, use_cpu=args.cpu,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        save_total_limit=1, save_only_model=True, optim="adamw_torch", report_to="none",
        seed=seed, data_seed=seed, **{warmup: args.warmup_ratio},
    )


def fit_best(args, split, seed, tokenizer, preprocess, directory, device):
    start = perf_counter()
    train = dataset_for(split.train, tokenizer, preprocess, args.max_length)
    val = dataset_for(split.val, tokenizer, preprocess, args.max_length)
    training_seconds = perf_counter() - start
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    weights = None
    if args.class_weight:
        counts = np.bincount(split.train["label"], minlength=len(LABELS))
        weights = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float32, device=device)
        logging.info("Class weights: %s", weights.tolist())

    def compute_metrics(result):
        return {"macro_f1": macro_f1(result.label_ids, result.predictions.argmax(axis=1))}

    model_path = directory / "best_model"
    best_score = -1
    history = []
    for lr in args.learning_rates:
        logging.info("Seed %d | learning rate %.2g", seed, lr)
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(LABELS),
            id2label=dict(zip(LABELS, LABEL_NAMES)), label2id=dict(zip(LABEL_NAMES, LABELS)),
        )
        with TemporaryDirectory(prefix="trainer-", dir=directory) as scratch:
            trainer = WeightedTrainer(
                model=model, args=training_args(args, scratch, seed, lr, args.fp16),
                train_dataset=train, eval_dataset=val, processing_class=tokenizer,
                data_collator=collator, compute_metrics=compute_metrics, class_weights=weights,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
            )
            start = perf_counter()
            trainer.train()
            training_seconds += perf_counter() - start
            score = float(trainer.state.best_metric)
            history.append({"learning_rate": lr, "validation_macro_f1": score,
                            "training_steps": trainer.state.global_step,
                            "epochs_completed": trainer.state.epoch})
            if score > best_score:
                best_score = score
                trainer.save_model(str(model_path))
            del trainer, model
        logging.info("Validation macro-F1 tại lr %.2g: %.4f", lr, score)
    logging.info("Learning rate tốt nhất: %.2g", max(history, key=lambda row: row["validation_macro_f1"])["learning_rate"])
    write_json(directory / "learning_rate_search.json", history)
    tokenizer.save_pretrained(model_path)
    save_preprocessor(model_path / "preprocessing.json", preprocess)
    return model_path, training_seconds


def run_split(args, split, seed, tokenizer, preprocess, device):
    directory = run_directory(args, args.run_name or args.model_name, split, seed)
    logging.info("%s / %s / fold %d / seed %d | train=%d val=%d test=%d", split.scenario,
                 split.held_out, split.fold, seed, len(split.train), len(split.val), len(split.test))
    model_path, seconds = fit_best(args, split, seed, tokenizer, preprocess, directory, device)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    @torch.inference_mode()
    def infer(texts):
        probabilities = []
        for index in range(0, len(texts), args.eval_batch_size):
            batch = [preprocess(text) for text in texts[index:index + args.eval_batch_size]]
            encoded = tokenizer(batch, truncation=True, max_length=args.max_length, padding=True,
                                pad_to_multiple_of=8 if args.fp16 else None,
                                return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=args.fp16):
                logits = model(**encoded).logits
            probabilities.append(logits.float().softmax(dim=-1).cpu().numpy())
        return np.concatenate(probabilities)

    metrics = save_evaluation(
        directory, split, infer(split.test["text"].tolist()), args.run_name or args.model_name,
        seed, args, training_seconds=seconds, inference_ms=benchmark(infer, split.test, args.split_seed),
        model_path=model_path, device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )
    logging.info("Test macro-F1=%.4f | %s", metrics["macro_f1"], directory)


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model-name", default="uitnlp/visobert")
    parser.add_argument("--run-name", help="Tên trong báo cáo, hữu ích khi model là đường dẫn cục bộ")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2025, 3407])
    parser.add_argument("--learning-rates", "--learning-rate", type=float, nargs="+", default=[2e-5])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--segment", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.run_name == "tfidf-lr":
        parser.error("Tên tfidf-lr được dùng cho baseline.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Các seed phải khác nhau.")
    if (min(args.batch_size, args.eval_batch_size, args.gradient_accumulation_steps, args.epochs,
            args.max_length, args.patience) < 1 or min(args.learning_rates) <= 0
            or not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0):
        parser.error("Kiểm tra lại batch size, epochs, learning rate, max length, warmup và patience.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(args.seeds) < 3:
        logging.warning("Có dưới ba seed. Dùng kết quả để chạy thử và bổ sung seed trước khi báo cáo.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.fp16 and device.type != "cuda":
        parser.error("--fp16 cần GPU CUDA.")
    if args.fp16 is None:
        args.fp16 = device.type == "cuda"
    if args.fp16 and args.max_length % 8:
        parser.error("--max-length phải chia hết cho 8 khi dùng fp16.")
    if args.segment is None:
        args.segment = "phobert" in args.model_name.lower()
    logging.info("Model=%s | thiết bị=%s | fp16=%s | tách từ=%s",
                 args.model_name, device, args.fp16, args.segment)

    data = load_data(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    preprocess = make_preprocessor(args, segment=args.segment)
    for split in make_splits(data, args):
        for seed in args.seeds:
            run_split(args, split, seed, tokenizer, preprocess, device)


if __name__ == "__main__":
    main()

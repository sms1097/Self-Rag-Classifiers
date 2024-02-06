import numpy as np
import torch

import pandas as pd
import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


MODEL = "distilbert-base-uncased"
BATCH_SIZE = 64


"""
Regression code comes form Medium article by La Javaness R&D
"""


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

    # Compute accuracy
    # Based on the fact that the rounded score = true score only if |single_squared_errors| < 0.5
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(
        single_squared_errors
    )

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


"""
Loading and creating the datasets
"""


def train_test_split(df, holdout_frac=0.1):
    test_df = df.sample(frac=holdout_frac)
    train_df = df[~df.index.isin(test_df.index)]

    valid_df = test_df.sample(frac=0.5)
    test_df = test_df[test_df.index.isin(valid_df.index)]

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "valid": Dataset.from_pandas(valid_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def load_tokens_data():
    ds = load_dataset("sms1097/self_rag_tokens_train_data")
    df = pd.DataFrame(ds['train'].to_dict())
    return df


def load_retrieval_ds():
    df = load_tokens_data()
    retrieval_df = df[~df["retrieval"].isna()]
    retrieval_df = retrieval_df[["instruction", "retrieval"]]
    retrieval_df["retrieval"] = retrieval_df["retrieval"].apply(
        lambda x: 1 if x == "[Retrieval]" else 0
    )
    retrieval_df.columns = ["text", "label"]
    retrieval_ds = train_test_split(retrieval_df)
    return retrieval_ds


def load_relevant_ds():
    df = load_tokens_data()
    relevant_df = df[~df.relevant.isna()]
    relevant_df["text"] = df.apply(
        lambda row: f"Instruction:\n{row['instruction']}\nContext:\n{row['doc']}",
        axis=1,
    )
    relevant_df = relevant_df[["text", "relevant"]]
    relevant_df.columns = ["text", "label"]
    relevant_ds = train_test_split(relevant_df)
    return relevant_ds


def load_support_ds():
    # Support
    support_map = {
        "[Fully supported]": 3.,
        "[Partially supported]": 2.,
        "[No support / Contradictory]": 1.,
    }

    df = load_tokens_data()
    support_df = df[~df.support.isna()]
    support_df["text"] = support_df.apply(
        lambda row: f"Context: {row['doc']}\nAnswer: {row['answer']}", axis=1
    )
    support_df["support"] = support_df["support"].map(support_map)
    support_df = support_df[["text", "support"]]
    support_df.columns = ["text", "label"]
    support_ds = train_test_split(support_df)
    return support_ds


def load_utility_ds():
    # Utility
    df = load_tokens_data()
    utility_df = df[~df["utility"].isna()]
    utility_df["text"] = df.apply(
        lambda row: f"Instruction: {row['instruction']}\nAnswer: {row['answer']}",
        axis=1,
    )
    utility_df["utility"] = utility_df["utility"].apply(
        lambda x: float(x.lstrip("[Utility:").rstrip("]"))
    )
    utility_df = utility_df[["text", "utility"]]
    utility_df.columns = ["text", "label"]
    utility_ds = train_test_split(utility_df)
    return utility_ds


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-4, log=True),
        # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32]),
    }


def main(target_column,run_tuning=False):
    # TODO: SMOTE would be nice on these unbalanced datasets
    if target_column == "retrieval":
        ds = load_retrieval_ds()
        num_labels = 2
    elif target_column == "relevant":
        ds = load_retrieval_ds()
        num_labels = 2
    elif target_column == "support":
        ds = load_support_ds()
        num_labels = 1
    elif target_column == "utility":
        ds = load_utility_ds()
        num_labels = 1
    else:
        raise Exception(f"unsupported task of value '{target_column}'")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=num_labels
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=num_labels
        )

    tokenized_ds = ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        target_column + "_model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1" if target_column in ["retrieval", "relevant"] else 'accuracy',
        push_to_hub=True,
    )

    # TODO: Run some hyperparameter tuning
    trainer_kwargs = dict(
        # model=None,
        model=model,
        # model_init=model_init,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=(
            compute_metrics
            if target_column in ["retrieval", "relevant"]
            else compute_metrics_for_regression
        ),
    )

    trainer = (
        Trainer(**trainer_kwargs)
        if target_column in ['retrieval', 'relevant']
        else RegressionTrainer(**trainer_kwargs)
    )


    if run_tuning:
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            n_trials=4
        )

        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

    trainer.train()
    trainer.eval_dataset=tokenized_ds["test"]
    print(trainer.evaluate())



if __name__ == "__main__":
    target_columns = ["utility"]

    for col in target_columns:
        main(col)

import re

import pandas as pd
from datasets import Dataset, load_dataset

HF_TOKEN = ""


def split_on_tokens(input_str):
    # Use regular expression to find all occurrences of text inside square brackets
    matches = re.finditer(r"\[.*?\]", input_str)

    # Split the input string based on the matches
    result = []
    start = 0
    for match in matches:
        result.append(input_str[start : match.start()])
        result.append(match.group())
        start = match.end()

    result.append(input_str[start:])

    return [x for x in result if x != ""]


def self_rag_output_parse(obs, instruction):
    doc, support_token, relevant_token = None, None, None
    split_obs = iter(split_on_tokens(obs))
    retrieval_token = next(split_obs)

    if retrieval_token == "[Retrieval]":
        doc = next(split_obs)
        relevant_token = next(split_obs)

    answer = next(split_obs)

    if relevant_token == "[Relevant]":
        support_token = next(split_obs)

    utility_token = next(split_obs)

    return {
        "instruction": instruction,
        "retrieval": retrieval_token,
        "doc": doc,
        "relevant": relevant_token,
        "answer": answer,
        "support": support_token,
        "utility": utility_token,
    }


if __name__ == "__main__":
    # loading in data
    data = load_dataset("selfrag/selfrag_train_data")
    df = pd.DataFrame(data["train"].to_dict())

    # creating some checks on overlap of retieval metrics
    df["retrieval_count"] = df["output"].apply(lambda x: x.count("[Retrieval]"))
    df["no_retrieval_count"] = df["output"].apply(lambda x: x.count("[No Retrieval]"))
    df["retrieval_and_no_retrieval"] = df["output"].apply(
        lambda x: "[Retrieval]" in x and "[No Retrieval]" in x
    )

    # Only using examples where there is one retireval/no_retrieval
    # This really simplifies the logic/data and ensures better quality
    df = df[
        ((df["retrieval_count"] == 1) & (df["no_retrieval_count"] == 0))
        | ((df["retrieval_count"] == 0) & (df["no_retrieval_count"] == 1))
    ]

    records = df.apply(
        lambda row: self_rag_output_parse(row["output"], row["instruction"]), axis=1
    )
    output_df = pd.DataFrame.from_records(records)
    output_df = output_df[
        output_df.utility.isin([f"[Utility:{i}]" for i in range(1, 6)])
    ]
    output_df.sample(frac=1)

    ds = Dataset.from_pandas(output_df, split="train")
    ds.push_to_hub(
        "sms1097/self_rag_tokens_train_data",
        token=HF_TOKEN,
    )

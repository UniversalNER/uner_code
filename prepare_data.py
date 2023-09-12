from datasets import Dataset, DatasetDict
from pathlib import Path
import re

import pandas as pd

def read_uner(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)

    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            # ignore comments
            if line.startswith('#'): continue
            tok_id, token, tag = line.split('\t')[:3]
            tokens.append(token)
            if "OTH" in tag or tag == "B-O":
                tag = "O"
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    # return token_docs, tag_docs

    train_dict = {"tokens": token_docs, "ner_tags": tag_docs}
    dataset = Dataset.from_dict(train_dict)
    return dataset



def read_multiple_datasets(dataset_dirs: list[Path]):
    all_datasets = {}
    all_stats = []
    for dataset_dir in dataset_dirs:
        print(dataset_dir)
        dataset_dict = {}
        for file_path in dataset_dir.glob("*.iob2"):
            print(file_path)
            subset, split = file_path.name.split("-ud-")
            lang, domain = subset.split("_")
            split = split.split(".")[0]
            print(subset, lang, domain, split)
            dataset = read_uner(file_path)
            dataset = dataset.map(
                lambda x: {
                    "num_tokens": len(x["tokens"]), 
                    "num_entities": len([v for v in x["ner_tags"] if v.startswith("B-")])
                }
            )
            dataset_dict[split] = dataset
            num_rows = dataset.num_rows
            num_tokens = sum(dataset["num_tokens"])
            num_entities = sum(dataset["num_entities"])
            
            all_stats.append([subset, lang, domain, split, num_rows, num_tokens, num_entities])
        all_datasets[subset] = DatasetDict(**dataset_dict)
    # all_datasets = DatasetDict(**all_datasets)
    df_stats = pd.DataFrame(all_stats, columns=["subset", "lang", "domain", "split", "docs", "tokens", "entities"])
    return all_datasets, df_stats


def summarize_datasets(all_datasets, df_stats):
    latex_code = df_stats.pivot_table(
        index=["lang", "domain",], columns="split", values=["docs", "tokens", "entities"],
        aggfunc="sum",
        margins=True
    ).style.to_latex(sparse_index=True)
    print("Overall stats")
    print(latex_code)


DATASET_DIR = "./"
if __name__ == "__main__":
    dataset_dirs = [
        p
        for p in Path(DATASET_DIR).glob("UNER_*")
        if p.is_dir()
        
    ]
    all_datasets, df_stats = read_multiple_datasets(dataset_dirs)

    summarize_datasets(all_datasets, df_stats)


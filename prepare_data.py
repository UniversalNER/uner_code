from datasets import Dataset
from pathlib import Path
import re

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

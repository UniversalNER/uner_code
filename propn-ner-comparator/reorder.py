import typer
from conllu import parse
from pathlib import Path
import hashlib

def remove_tokens_with_non_int_ids(sentences: list):
    for sentence in sentences:
        yield sentence.filter(id=lambda x: type(x) is int)

def hash(input: str):
    return hashlib.sha3_512(input.encode()).digest()

def sentence_to_text(sent):
    return ' '.join(map(lambda x: x['form'].lower(), sent))

def sentence_to_hash(sent):
    return hash(sentence_to_text(sent))

def reorder(conllu_file: Path, iob_file: Path, ordered_file: Path):

    conllu_sentences = parse(conllu_file.read_text())
    iob_sentences = parse(iob_file.read_text())

    conllu_sentences = remove_tokens_with_non_int_ids(conllu_sentences)
    iob_sentences = remove_tokens_with_non_int_ids(iob_sentences)

    id2iob = {sentence_to_hash(s): s for s in iob_sentences}

    ordered_sentences = []

    for s in conllu_sentences:
        h = sentence_to_hash(s)
        try:
            ordered_sentences.append(id2iob[h].serialize())
        except KeyError:
            print(s, sentence_to_text(s))

    ordered_file.write_text('\n\n'.join(ordered_sentences))


if __name__ == "__main__":
    typer.run(reorder)

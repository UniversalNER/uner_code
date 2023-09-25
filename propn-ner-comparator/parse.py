import typer
from conllu import parse
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score

def parse_compare(conllu_file: Path, iob_file: Path):
    
    conllu_sentences = parse(conllu_file.read_text())
    iob_sentences = parse(iob_file.read_text())

    n_tokens = 0
    n_misaligned = 0

    propn_arr = []
    tagged_arr = []

    for conllu_s, iob_s in zip(conllu_sentences, iob_sentences):
        for conllu_tok, iob_tok in zip(conllu_s, iob_s):
            n_tokens += 1
            if str(conllu_tok) != str(iob_tok):
                n_misaligned += 1
            else:
                is_propn = (conllu_tok['upos'] == 'PROPN')
                is_tagged = (iob_tok['lemma'] != 'O')

                propn_arr.append(is_propn)
                tagged_arr.append(is_tagged)


        
    print(f"Tokens: {n_tokens}, misaligned: {n_misaligned} / {(n_misaligned/n_tokens):.2%}")
    print(confusion_matrix(propn_arr, tagged_arr))

    tn, fp, fn, tp = confusion_matrix(propn_arr, tagged_arr).ravel()
    all_valid_tokens = tn + tp + fn + fp

    print()
    print(f"F1 score: {f1_score(propn_arr, tagged_arr)}")
    print(f"PROPN w\\o NER: {fp} / {all_valid_tokens} ({(fp) / all_valid_tokens:.2%})")
    print(f"NER w\\o PROPN: {fn} / {all_valid_tokens} ({(fn) / all_valid_tokens:.2%})")


if __name__ == "__main__":
    typer.run(parse_compare)

# PROPN vs NER tag comparator

A quick script that compares the coverage of tokens tagged with PROPN in
Universal Dependencies with NER tags as part of the UniversalNER project.

## Installation

To install the dependencies, run

    $ pip install -r requirements.txt

Alternatively, the dependencies are also managed via [Poetry](https://python-poetry.org/docs/).

    $ poetry install

## Execution

The script expects two files as its argument: the `.conllu` from Universal
Dependencies as the first one and `.iob2` from UniversalNER as the second one.

To find the coverage between the two files one can run something like

    $ poetry run python parse.py data/UD_Chinese-PUD/zh_pud-ud-test.conllu data/UNER_Chinese-PUD/zh_pud-ud-test.iob2

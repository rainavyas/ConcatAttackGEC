# Task
Universal concatenation adversarial attack on Grammatical Error Correction system.

Grammatical Error Correction (GEC) systems can be used a measure of a candidate's fluency. A GEC systems that makes no changes to an input text (i.e. no corrections) suggests that the input text is grammaticaly perfect. In an assessment context, a candidate may attempt mal-practice through performing an adversarial attack that modifies their incorrect input text in a manner that fools the GEC system into making no corrections, i.e. falsely suggesting the input text is perfect.

In this work, a universal concatentation adversarial attack is considered, where the aim is to find a single _N_ length sequence of words that can be appeded to any input sequence to fool the GEC system into making as few corrections as possible. The impact of the adversarial attack is measured using a count of _edits_ between the original input and GEC output text: the greater the average reduction in edits, the more effective the universal adversarial attack.

Here, experiments are performed on the [FCE public dataset](https://ilexir.co.uk/datasets/index.html) and a T5 based GEC system trained by _vennify_, available [here](https://huggingface.co/vennify/t5-base-grammar-correction?).

# Requirements

Clone this repository.

## Install with PyPI

`pip install happytransformer`
`pip install errant`

# Experimental Results

The universal attack phrase is learnt on the training set and the impact of the attack here are reported on the test set.

|N|Average Edits|
|---------------|-------------|
0 | 1.70 |
1 | 1.74 |
2 | 1.89 |
3 | 0.82 |
4 | 0.91 |
5 | 1.85 |
6 | 1.01 |

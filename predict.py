'''
Generate model predictions

Input file:
ID1 Sentence1
ID2 Sentence2
.
.
.

Output file:
ID1 Sentence1
ID2 Sentence2
.
.
.

If an adversarial sequence is appended at the input, then two files are created:
adv_with_adv_not_removed.pred
adv_with_adv_removed.pred

i.e. the second file has the adversarial sequence removed from the output prediction sentences
'''

import sys
import os
import argparse
from happytransformer import HappyTextToText, TTSettings
import torch
from gec_tools import get_sentences, correct


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('OUT_BASE', type=str, help='Path to corrected output data - pass only base name, e.g. beam1_N4 or no_attack')
    commandLineParser.add_argument('--phrase', type=str, default='', help='Universal adversarial phrase')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    device = torch.device('cpu')

    # Load Model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    model.device = torch.device('cpu')
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load input sentences
    identifiers, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence
    corrections = []
    if args.phrase != '':
        corrections_adv_removed = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        if args.phrase != '':
            sent = sent + ' ' + args.phrase + '.'
        correction = correct(model, sent, gen_args)
        corrections.append(correction)
        if args.phrase != '':
            correction_adv_removed = correction.replace(f' {args.phrase}.', '')
            corrections_adv_removed.append(correction_adv_removed)        
    assert len(corrections) == len(identifiers), "Number of ids don't match number of predictions"

    # Save predictions
    if args.phrase != '':
        file2 = f'{args.OUT_BASE}_with_adv_removed.pred'
        file1 = f'{args.OUT_BASE}_with_adv_not_removed.pred'
        with open(file2, 'w') as f:
            for id, sentence in zip(identifiers, corrections_adv_removed):
                f.write(f'{id} {sentence}\n')
    else:
        file1 = f'{args.OUT_BASE}.pred'

    with open(file1, 'w') as f:
        for id, sentence in zip(identifiers, corrections):
            f.write(f'{id} {sentence}\n')

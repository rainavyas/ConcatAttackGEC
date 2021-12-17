'''
Evaluate universal adversarial attack.

Evaluate by counting average number of edits between original input
(with attack phrase) and  GEC model prediction
'''

import sys
import os
import argparse
from happytransformer import HappyTextToText, TTSettings
import torch
from gec_tools import get_sentences, correct, count_edits
from statistics import mean, stdev

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('--phrase', type=str, default='', help='Universal adversarial phrase')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 
    
    # Load Model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    model.device = torch.device('cpu')
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load input sentences
    _, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence
    edit_counts = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        attack_sent = sent + ' ' + args.phrase
        correction = correct(model, attack_sent, gen_args)
        edit_counts.append(count_edits(attack_sent, correction))
    
    edits_mean = mean(edit_counts)
    edits_std = stdev(edit_counts)
    print(f'Mean: {edits_mean}\t Std: {edits_std}')


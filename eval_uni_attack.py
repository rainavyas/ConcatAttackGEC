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
import matplotlib.pyplot as plt

def print_stats(edits, name):
    print()
    print(f'{name}: {len(edits)} samples')
    edits_mean = mean(edits)
    edits_std = stdev(edits)
    print(f'\nMean: {edits_mean}\t Std: {edits_std}')
    print()

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('FIG', type=str, help='Where to save histogram plot')
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

    # Correction (prediction) for each input sentence and associated number of edits
    edit_counts = []
    edit_counts_with_attack = []
    for i, sent in enumerate(sentences):

        print(f'On {i}/{len(sentences)}')
        sent_with_attack = sent + ' ' + args.phrase

        correction = correct(model, sent, gen_args)
        correction_with_attack = correct(model, sent_with_attack, gen_args)

        edit_counts.append(count_edits(sent, correction))
        edit_counts_with_attack.append(count_edits(sent_with_attack, correction_with_attack))
    

    # Print stats for all samples
    print_stats(edit_counts_with_attack, 'ALL')

    # Print stats for samples filtered by number of actual errors (from no attack)
    thresholds = [0, 1, 2, 3, 4, 5]
    for thresh in thresholds:
        edits_imperfect = [b for a,b in zip(edit_counts, edit_counts_with_attack) if a>thresh]
        name = f'Filtered >{thresh}'
        print_stats(edits_imperfect, name)
    
    # Plot histogram of edit count distribution before and after attack
    plt.hist(edit_counts, bins=10, alpha=0.5, label='No Attack')
    plt.hist(edit_counts_with_attack, bins=10, alpha=0.5, label='With Attack')
    plt.xlabel("Edits")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(args.FIG, bbox_inches='tight')








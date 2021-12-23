'''
Input: Training samples
Output: Filtered training samples

where the filtering requirement is that edits > k, 
i.e. we want to only keep training samples which have more than
k grammatical errors.
'''

import sys
import os
import argparse
from happytransformer import HappyTextToText, TTSettings
import torch
from gec_tools import get_sentences, correct, count_edits

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('OUT', type=str, help='Path to output filtered data')
    commandLineParser.add_argument('--thresh', type=int, default=1, help='Filter >thresh edits')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/filter.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load Model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    model.device = torch.device('cpu')
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load input sentences
    ids, sentences = get_sentences(args.IN)

    # Correction (prediction) for each input sentence and associated number of edits
    edit_counts = []
    for i, sent in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        correction = correct(model, sent, gen_args)
        edit_counts.append(count_edits(sent, correction))
    
    kept_ids = [id for id,e in zip(ids, edit_counts) if e>args.thresh]
    kept_sentences = [s for s,e in zip(sentences, edit_counts) if e>args.thresh]

    # Save filtered
    with open(args.OUT, 'w') as f:
        for id, sentence in zip(kept_ids, kept_sentences):
            f.write(f'{id} {sentence}\n')
    

    

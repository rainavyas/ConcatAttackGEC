'''
At input expect a SOURCE and TARGET file:

ID1 text
ID2 text
.
.
.

Output the errant edit type distribution from SOURCE to TARGET file

This scripts was written to find the edit types from original
.inc file and reference .corr file.
'''
import sys
import os
import argparse
from align_preds import get_sentences_dict
from gec_tools import return_edits
from eval_uni_attack import update_edit_types
from collections import defaultdict

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('SOURCE', type=str, help='Path to source data')
    commandLineParser.add_argument('TARGET', type=str, help='Path to target data')
    commandLineParser.add_argument('EDIT_TYPE', type=str, help='Path to save edit type information')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_error_dist.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 

    # Get sentences and align
    source_id2text = get_sentences_dict(args.SOURCE)
    target_id2text = get_sentences_dict(args.TARGET)

    source_sens = []
    target_sens = []
    for id, text in target_id2text.items():
        try:
            source_sens.append(source_id2text[id]+'\n')
            target_sens.append(target_id2text[id]+'\n')
        except:
            pass
    assert len(source_sens) == len(target_sens), "Mismatch in num items"


    # Get the edit types
    edit_types = defaultdict(int)
    for i, (s, t) in enumerate(zip(source_sens, target_sens)):
        print(f'On {i}/{len(source_sens)}')
        edits = return_edits(s, t)
        update_edit_types(edits, edit_types)

    # Save edit type distribution to file
    texts = ['Type Count']
    for edit_type in sorted(list(edit_types.keys())):
        texts.append(f'\n{edit_type} {edit_types[edit_type]}')
    with open(args.EDIT_TYPE, 'w') as f:
            f.writelines(texts)
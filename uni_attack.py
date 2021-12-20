'''
Perform concatenation adversarial attack on 
GEC system, with aim of finding universal adversarial phrase
that minimises average number of edits between original and 
predicted gec sentence.
'''
import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from happytransformer import HappyTextToText, TTSettings
import json
from datetime import date
from statistics import mean

def get_avg(model, sentences, attack_phrase, gen_args):
    edit_counts = []
    for sent in sentences:
        sent = sent + ' ' + attack_phrase
        correction = correct(model, sent, gen_args)
        edit_counts.append(count_edits(sent, correction))
    return mean(edit_counts)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('--prev_attack', type=str, default='', help='greedy universal attack phrase')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')
    commandLineParser.add_argument('--search_size', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--start', type=int, default=0, help='Vocab batch number')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uni_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load Model
    model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    model.device = torch.device('cpu')
    gen_args = TTSettings(num_beams=5, min_length=1)

    # Load input sentences
    _, sentences = get_sentences(args.IN, num=args.num_points)

    # Get list of words to try
    with open(args.VOCAB, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = args.start*args.search_size
    test_words = test_words[start_index:start_index+args.search_size]

    # Add blank word at beginning of list
    # test_words = ['']+test_words

    # Initialise empty log file
    with open(args.LOG, 'w') as f:
        f.write("Logged on "+ str(date.today()))

    best = ('none', 1000)
    for word in test_words:
        attack_phrase = args.prev_attack + ' ' + word
        edits_avg = get_avg(model, sentences, attack_phrase, gen_args)
        # print(word, edits_avg) # temp debug

        if edits_avg < best[1]:
            best = (word, edits_avg)
            # Write to log
            with open(args.LOG, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)

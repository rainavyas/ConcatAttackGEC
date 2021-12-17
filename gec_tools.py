import random
import errant

def get_sentences(data_path, num=None):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    if num is not None:
        random.seed(10)
        lines = random.shuffle(lines)
        lines = lines[:num]
    texts = [' '.join(l.rstrip('\n').split()[1:]) for l in lines]
    ids = [l.rstrip('\n').split()[0] for l in lines]
    return ids, texts

def correct(model, sentence, gen_args):
    correction_prefix = "grammar: "
    sentence = correction_prefix + sentence
    result = model.generate_text(sentence, gen_args)
    return result.text

def count_edits(input, prediction):
    '''
    Count number of edits
    '''
    annotator = errant.load('en')
    alignment = annotator.align(input, prediction)
    edits = annotator.merge(alignment)
    return len(edits)
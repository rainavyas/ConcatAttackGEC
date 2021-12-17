def get_sentences(data_path, num=None):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    texts = [' '.join(l.rstrip('\n').split()[1:]) for l in lines]
    ids = [l.rstrip('\n').split()[0] for l in lines]
    if num is not None:
        texts = texts[:num]
        ids = ids[:num]
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
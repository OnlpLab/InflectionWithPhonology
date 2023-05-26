from DataRelatedClasses.Vocabs.EditVocab import EditVocab
import numpy as np

def remove_pipe(string):
    string = string.strip('|')
    string = string.strip()
    try:
        if string[-3] == '|':
            string = string[:-3]
    except IndexError:
        pass
    return string


def action2string(actions, vocab: EditVocab):
    stringified_actions = [vocab.act.i2w[a] for a in actions]
    if '$' in vocab.act.i2w.keys():
        return ', '.join(stringified_actions)
    else:
        return ' '.join(stringified_actions)

def feats2string(pos, feats, vocab):
    if pos:
        pos_str = vocab.pos.i2w[pos] + ';'
    else:
        pos_str = ''
    return  pos_str + ';'.join(vocab.feat.i2w[f] for f in feats)

def generate_random_indices_in_range(number_of_samples: int) -> set:
    return set(np.random.choice(8000, number_of_samples, False)) # the seed is set at the run's beginning
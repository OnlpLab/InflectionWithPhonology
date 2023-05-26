#-*- coding: utf-8 -*-
from itertools import chain
from more_itertools import split_at
from typing import Callable, List, Tuple, Union

from defaults import ALIGN_SYMBOL

def split_iterable(iterable, delimiter):
    return split_at(iterable, lambda e: e == delimiter)

def _join_iterable(iterable, delimiter):
    # Inserts delimiters between elements of some iterable object.
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

# The inverse of split_iterable
def join_iterable(iterable, delimiter='$') -> List:
    return list(chain(*_join_iterable(iterable, (delimiter,))))


def cls_align(pairs: Union[List[Tuple[str, str]], Tuple], multiword=True, **kwargs): # todo oracle: multiword=False
    return multiword_align(pairs, _cls_align, multiword, **kwargs)

def multiword_align(pairs: List[Tuple[str, str]], _align: Callable, multiword, **kwargs):
    if multiword:
        alignedpairs = []
        for ins, outs in pairs:

            if type(ins) == str:
                if ins.count(' ') == outs.count(' '):
                    # assume multiword expression
                    aligned_ins, aligned_outs = [], []
                    for subins, subouts in zip(ins.split(' '), outs.split(' ')):
                        aligned_subins, aligned_subouts = _align(subins, subouts)
                        aligned_ins.append(aligned_subins)
                        aligned_outs.append(aligned_subouts)
                    aligned_ins, aligned_outs = ' '.join(aligned_ins), ' '.join(aligned_outs)
                else:
                    aligned_ins, aligned_outs = _align(ins, outs)
            else: # type(pairs[0][0]) == list
                SPACE_CHARACTER = kwargs['space_character']
                if SPACE_CHARACTER is None:
                    assert outs.count(' ') == 0
                if ins.count(SPACE_CHARACTER) == outs.count(SPACE_CHARACTER):
                    # assume multiword expression
                    aligned_ins, aligned_outs = [], []
                    ins_iterator, outs_iterator = split_iterable(ins, SPACE_CHARACTER), split_iterable(outs, SPACE_CHARACTER)
                    for subins, subouts in zip(ins_iterator, outs_iterator):
                        aligned_subins, aligned_subouts = _align(subins, subouts, [ALIGN_SYMBOL])  # fixed the "to-do oracle [align_symbol]"

                        aligned_ins.append(aligned_subins)
                        aligned_outs.append(aligned_subouts)
                    aligned_ins, aligned_outs = join_iterable(aligned_ins, SPACE_CHARACTER), join_iterable(aligned_outs, SPACE_CHARACTER)
                else:
                    aligned_ins, aligned_outs = _align(ins, outs, [ALIGN_SYMBOL])

            alignedpairs.append((aligned_ins, aligned_outs))
    else:
        alignedpairs = [_align(*p) for p in pairs]
    return alignedpairs

def _cls_align(ins, outs, align_symbol: Union[str, List[str]]=ALIGN_SYMBOL):
    len_ins  = len(ins)
    len_outs = len(outs)
    LCSuff = [[0 for _ in range(len_outs + 1)] for _ in range(len_ins + 1)]
    cls_length = 0
    pointer = 0, 0
    for i in range(len_ins + 1):
        for j in range(len_outs + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] == 0
            elif ins[i-1] == outs[j-1]:
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                if LCSuff[i][j] > cls_length:
                    cls_length = LCSuff[i][j]
                    pointer = i-1, j-1
            else:
                LCSuff[i][j] = 0

    aligned_ins, aligned_outs = ins, outs
    # cls'es should be aligned, the rest aligned and padded
    # pad from the left
    offset = pointer[0] - pointer[1]
    if offset > 0:
        # the cls starts later in ins, and so outs need to be padded.
        aligned_outs = align_symbol * offset + aligned_outs
    elif offset < 0:
        aligned_ins = align_symbol * abs(offset) + aligned_ins

    # pad from the right
    length_diff = len_ins - len_outs - offset
    if length_diff > 0:
        aligned_outs += align_symbol * length_diff
    elif length_diff < 0:
        aligned_ins += align_symbol * abs(length_diff)

    return aligned_ins, aligned_outs


def smart_align(pairs, align_symbol=ALIGN_SYMBOL,
                iterations=150, burnin=5, lag=1, mode='crp', **kwargs):
    import align
    return align.Aligner(pairs,
                         align_symbol=align_symbol,
                         iterations=iterations,
                         burnin=burnin,
                         lag=lag,
                         mode=mode).alignedpairs

def dumb_align(pairs, align_symbol=ALIGN_SYMBOL, multiword=True, **kwargs):
    def _dumb_align(ins, outs):
        length_diff = len(ins) - len(outs)
        if length_diff > 0:
            outs += align_symbol * length_diff
        elif length_diff < 0:
            ins += align_symbol * abs(length_diff)
        return ins, outs
    return multiword_align(pairs, _dumb_align, multiword)

if __name__ == "__main__":
    
    def seq_of_pairs_unicode(l):
        return ', '.join('({}, {})'.format(*p) for p in l)
        
    pairs = (('walk', 'walked'), ('fliegen', 'flog'), ('береза', 'берёз'), ('집', '집'),
             ('sing', 'will sing'), ('белый хлеб', 'белого хлеба'))
    dumb_targets_nomulti = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), ('береза', 'берёз~'),
                            ('집', '집'), ('sing~~~~~', 'will sing'), ('белый хлеб~~', 'белого хлеба'))
    dumb_targets = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), ('береза', 'берёз~'),
                    ('집', '집'), ('sing~~~~~', 'will sing'), ('белый~ хлеб~', 'белого хлеба'))
    cls_targets = (('walk~~', 'walked'), ('fliegen', 'flog~~~'), ('береза', 'берёз~'), ('집', '집'),
               ('~~~~~sing', 'will sing'), ('белый~ хлеб~', 'белого хлеба'))

    alignment, aligned_pairs, targets = 'CLS', cls_align(pairs), cls_targets
    print('Alignment: {}'.format(alignment))
    print('Pairs:         {}'.format(seq_of_pairs_unicode(pairs)))
    print('Aligned pairs: {}'.format(seq_of_pairs_unicode(aligned_pairs)))
    print('Targets:       {}'.format(seq_of_pairs_unicode(targets)))
    for a, t in zip(aligned_pairs, targets):
        assert a == t, 'Mismatch: {} and {}'.format(a, t)
    print('All matches.')
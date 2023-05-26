from itertools import chain
import re
from typing import List, Union, Tuple, Iterable

from PhonologyConverter.g2p_config import idx2feature, feature2idx, p2f_dict, f2p_dict, langs_properties, punctuations

def joinit(iterable: Iterable, delimiter):
    # Inserts delimiters between elements of some iterable object.
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x

def tuple_of_phon_tuples2phon_sequence(tupleOfTuples: Iterable[Tuple[str]], use_separator=True) -> List[str]:
    return list(chain(*(joinit(tupleOfTuples, ('$',) ) if use_separator else tupleOfTuples)))

def normalize_dollars(sequence: List[str], verbose = False) -> List[str]:
    """
    Takes a sequence of phonological features, separated by '$'s, and normalizes the sequence to the closest valid sequence.
    The normalizing includes removal of any unnecessary '$'s.
    :param sequence: e.g. ['1', '2', '3', '$', '4', '5', 'NA', '$', ...], possibly invalid.
    :param verbose: prints a corresponding message
    :return: The sequence without the unnecessary '$' signs.
    """
    while sequence[0] == '$' and sequence != []: del sequence[0]
    while sequence[-1] == '$' and sequence != []: del sequence[-1]

    comma_replaced_sequence = re.sub(r'\$,[$,]+', '$,', ','.join(sequence))
    result_sequence = comma_replaced_sequence.split(',')

    if verbose:
        if result_sequence == sequence: print(f"Sequence was valid, no corrections were made")
        else: print("Sequence wasn't valid, some corrections were made")

    return result_sequence


class LanguageSetup:
    """
    Stores the letters and phonemes of the language. Further logic includes converting words to phonemes and vice versa.
    Note: the class is implemented to fit to Georgian, Russian and several Indo-European languages. For languages with more complex phonology,
    this class might need to be extended/be inherited from.
    """
    def __init__(self, lang_name: str, graphemes2phonemes: dict, max_phoneme_size: int,
                 phon_use_attention: bool, manual_word2phonemes=None, manual_phonemes2word=None):
        self._name = lang_name
        self._graphemes2phonemes = graphemes2phonemes
        self._graphemes2phonemes.update(dict(zip(punctuations, punctuations)))
        self._phonemes2graphemes = {v:k for k, v in self._graphemes2phonemes.items()} # _graphemes2phonemes.reverse

        self._alphabet = list(self._graphemes2phonemes.keys()) # the graphemes of the language
        self._phonemes = list(self._graphemes2phonemes.values())

        self._manual_word2phonemes = manual_word2phonemes
        self.manual_phonemes2word = manual_phonemes2word

        self.max_phoneme_size = max_phoneme_size
        self.phon_use_attention = phon_use_attention

    def get_lang_name(self): return self._name
    def get_lang_alphabet(self): return self._alphabet
    def get_lang_phonemes(self): return self._phonemes

    def word2phonemes(self, word: str, mode: str, use_separator=True) -> Union[List[List[int]], List[str]]:
        """
        Convert a word (sequence of graphemes) to a list of phoneme tuples.
        :param word: word
        :param mode: can be either 'features' or 'phonemes' (see more above).
        :param use_separator: if mode == 'features', then if True, separating between the features tuples with '$'.
        :return: ((,,), (,,), (,,), ...) or list(*IPA symbols*)
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"

        word = word.casefold() # lower-casing
        graphemes = list(word) # beware of Niqqud-like symbols

        if self._manual_word2phonemes:
            phonemes = self._manual_word2phonemes(graphemes)
        else:
            phonemes = [self._graphemes2phonemes[g] for g in graphemes]

        if mode=='phonemes':
            return phonemes
        else: # mode=='features'
            features=[]
            for p in phonemes:
                feats = [str(feature2idx[e]) for e in p2f_dict[p]]
                feats.extend(['NA']*(self.max_phoneme_size-len(feats)))
                if self.phon_use_attention:
                    feats.append(p)
                features.append(tuple(feats))
            features = tuple_of_phon_tuples2phon_sequence(features, use_separator=use_separator)
            return features

    def _phonemes2word(self, phonemes: Union[List[str], Tuple[str], List[List[str]]], mode: str) -> str:
        """
        Convert a list of phoneme tuples to a word (sequence of graphemes)
        :param phonemes: [(,,), (,,), (,,), ...] or a list of IPA symbols
        :param mode: can be either 'features' or 'phonemes' (see more above).
        :return: word
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"

        if mode=='phonemes':
            if self.manual_phonemes2word:
                graphemes = self.manual_phonemes2word(phonemes)
            else:
                graphemes = [self._phonemes2graphemes[p] for p in phonemes]
        else: # mode=='features'
            if self.phon_use_attention:
                phoneme_tokens = [f_tuple[-1] for f_tuple in phonemes]
            else:
                phoneme_tokens = []
                for f_tuple in phonemes:
                    f_tuple = tuple([idx2feature[int(i)] for i in f_tuple if i != 'NA'])
                    p = f2p_dict.get(f_tuple)
                    if p is None or p not in self._phonemes: p = "*" # the predicted bundle is illegal or doesn't exist in this language
                    phoneme_tokens.append(p)
            graphemes = self._phonemes2word(phoneme_tokens, 'phonemes')
        return ''.join(graphemes)

    def phonemes2word(self, sequence: Union[List[str], Tuple[str]], mode: str, normalize = False) -> str:
        """
        Wrapper for _phonemes2word.
        sequence is list, of either features (['1', '2', '3', '$', '4', '5', 'NA', '$', ...]),
        phonemes (['p', 't', 'o:', ...]), or combined (['1', '2', '3', 'p', '$', '4', '5', 'NA', 'o:', '$', ...])
        """
        assert mode in {'features', 'phonemes'}, f"Mode {mode} is invalid"
        if type(sequence) == tuple: sequence = list(sequence)

        if mode == 'features':
            if not sequence: return '' # edge case: empty prediction

            if normalize:
                sequence = normalize_dollars(sequence)

            comma_separated_phonemes_list = ','.join(sequence).split(',$,')
            if self.manual_phonemes2word and self.phon_use_attention:
                new_sequence = self._phonemes2word([e.split(',')[-1] for e in comma_separated_phonemes_list], mode='phonemes')
            else:
                new_sequence = self._phonemes2word([p.split(',') for p in comma_separated_phonemes_list], mode='features')
        else: # mode=='phonemes'
            new_sequence = self._phonemes2word(sequence, mode='phonemes')

        return new_sequence

    @classmethod
    def create_phonology_converter(cls, language: str, phon_use_attention = False):
        # Calculating and instantiating the dynamic objects
        max_feat_size = max([len(p2f_dict[p]) for p in langs_properties[language][0].values() if p in p2f_dict])  # composite phonemes aren't counted in that list

        return cls(language,
                   langs_properties[language][0],
                   max_feat_size,
                   phon_use_attention,
                   langs_properties[language][1],
                   langs_properties[language][2])

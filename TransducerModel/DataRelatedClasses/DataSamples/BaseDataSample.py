from typing import List

from DataRelatedClasses.utils import feats2string, remove_pipe
from defaults import BEGIN_WORD, END_WORD, SPECIAL_CHARS
from DataRelatedClasses.Vocabs.VocabBox import VocabBox
from PhonologyConverter.languages_setup import LanguageSetup

def assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str):
    # encode features as integers
    # POS feature treated separately
    row_items = [input_str, output_str, in_feats_str, out_feats_str]
    for item in row_items:
        assert isinstance(item, str), item

    assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
    assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)

class BaseDataSample(object):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, word_phonological,
                 out_pos, out_feats, out_feat_str, tag_wraps, vocab, input_phonemes=None, input_phonemes_str=None):
        self.vocab = vocab  # vocab of unicode strings

        self.lemma = lemma  # list of encoded lemma characters
        self.lemma_str = lemma_str  # unicode string

        self.word = word  # encoded word
        self.word_str = word_str  # unicode string
        self.word_phonological = word_phonological

        self.in_pos = in_pos  # encoded input pos feature
        self.in_feats = in_feats  # set of encoded input features

        self.out_pos = out_pos  # encoded output pos feature
        self.out_feats = out_feats  # set of encoded output features

        self.in_feat_str = in_feat_str  # original serialization of features, unicode
        self.out_feat_str = out_feat_str  # original serialization of features, unicode

        # new serialization of features, unicode
        self.in_feat_repr = feats2string(self.in_pos, self.in_feats, self.vocab)
        self.out_feat_repr = feats2string(self.out_pos, self.out_feats, self.vocab)

        self.tag_wraps = tag_wraps  # were lemma / word wrapped with word boundary tags '<' and '>'?

        self.phonemes = input_phonemes
        self.phonemes_str = input_phonemes_str # a tuple of the phonemes/features. Needed for ED eval on the dev

    def __repr__(self):
        return f'Input: {self.lemma_str},Features: {self.in_feat_repr}, Output: {self.word_str}, ' \
               f'Features: {self.out_feat_repr}, Wraps: {self.tag_wraps}'

    @classmethod
    def from_row(cls, vocab: VocabBox, tag_wraps: str, verbose, row: List[str], sigm2017format=True,
                 phonology_converter:LanguageSetup=None, use_self_attention=False):
        if sigm2017format:
            input_str, in_feats_str, output_str, out_feats_str = row
            input_str = remove_pipe(input_str)
            output_str = remove_pipe(output_str)
        else:
            in_feats_str, input_str, out_feats_str, output_str = row
        feats_delimiter = u';'

        assert_inputs_are_valid(input_str, output_str, in_feats_str, out_feats_str)

        if phonology_converter is None:
            # encode input characters
            input = [vocab.char[c] for c in input_str]  # .split()]
            # encode word
            word = vocab.word[output_str]  # .replace(' ','')]
            word_phonological = None
            input_phonemes, input_phonemes_str = None, None
        else:
            input_features = phonology_converter.word2phonemes(input_str, 'features', use_separator = not use_self_attention)
            # encode input characters
            input = [vocab.char[c] for c in input_features]  # .split()]

            if use_self_attention:
                # compute the indices' representation of input_phonemes, and use it later in the encoding if use_self_attention
                input_phonemes_str = phonology_converter.word2phonemes(input_str, 'phonemes')
                input_phonemes = [vocab.char[c] for c in input_phonemes_str]
                # we need the phonemic representation of the output, without features
                output_features = tuple(phonology_converter.word2phonemes(output_str, 'phonemes'))
            else:
                input_phonemes, input_phonemes_str = None, None
                output_features = tuple(phonology_converter.word2phonemes(output_str, 'features'))

            # encode word
            word = vocab.word[output_features]  # .replace(' ','')]
            word_phonological = [vocab.char[c] for c in output_features]

        in_feats = in_feats_str.split(feats_delimiter)
        out_feats = out_feats_str.split(feats_delimiter)

        in_feats = [vocab.feat[f] for f in set(in_feats)]
        out_feats = [vocab.feat[f] for f in set(out_feats)]

        in_pos, out_pos = None, None

        # wrap encoded input with (encoded) boundary tags
        if tag_wraps == 'both':
            input = [BEGIN_WORD] + input + [END_WORD]
            if use_self_attention:
                input_phonemes = [BEGIN_WORD] + input_phonemes + [END_WORD]
        elif tag_wraps == 'close':
            input = input + [END_WORD]

        # print features and input at a high verbosity level
        if verbose == 2:
            # print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, output_str, input_str, pos, feats)
            print(f'POS & features from {input_str}, {output_str}: {in_pos}, {in_feats} --> {out_pos}, {out_feats}')
            print(f'input encoding: {input}')

        return cls(input, input_str, in_pos, in_feats, in_feats_str, word, output_str, word_phonological,
                   out_pos, out_feats, out_feats_str, tag_wraps, vocab,
                   input_phonemes=input_phonemes, input_phonemes_str=input_phonemes_str)

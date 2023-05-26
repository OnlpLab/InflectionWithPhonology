from DataRelatedClasses.Vocabs.Vocab import Vocab
from defaults import UNK, UNK_CHAR

class VocabBox(object):
    def __init__(self, acts, encoding):

        self.w2i_acts = acts
        self.act = Vocab(acts, encoding=encoding)

        # number of special actions
        self.number_specials = len(self.w2i_acts)

        # special features
        w2i_feats = {UNK_CHAR: UNK}
        self.feat = Vocab(w2i_feats, encoding=encoding)

        self.pos = self.feat

        self.feat_type = self.feat

        # use one set of indices for acts and chars
        self.char = self.act
        print('VOCAB will use same indices for actions and chars.')

        # encoding of words
        self.word = Vocab(encoding=encoding)

        # training set cut-offs
        self.act_train = None
        self.feat_train = None
        self.pos_train = None
        self.char_train = None
        self.feat_type_train = None

    def __repr__(self):
        return f'VocabBox (act, feat, pos, char, feat_type) with the following ' \
               f'special actions: {self.w2i_acts}'

    def train_cutoff(self):
        # store indices separating training set elements
        # from elements encoded later from unseen samples
        self.act_train = len(self.act)
        self.feat_train = len(self.feat)
        self.pos_train = len(self.pos)
        self.char_train = len(self.char)
        self.feat_type_train = len(self.feat_type)


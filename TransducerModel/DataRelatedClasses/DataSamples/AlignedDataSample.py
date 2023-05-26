from DataRelatedClasses.DataSamples.BaseDataSample import BaseDataSample
from DataRelatedClasses.utils import action2string

class AlignedDataSample(BaseDataSample):
    # data sample with encoded oracle actions derived from character alignment of lemma and word
    def set_actions(self, actions, aligned_lemma, aligned_word):
        self.actions = actions              # list of indices
        # serialization of actions as unicode string
        self.act_repr = action2string(self.actions, self.vocab)
        self.aligned_lemma = aligned_lemma  # unicode string
        self.aligned_word = aligned_word    # unicode string

    def __repr__(self):
        return f'Input: {self.lemma_str},Features: {self.in_feat_repr}, Output: {self.word_str}, ' \
               f'Features: {self.out_feat_repr}, Actions: {self.act_repr}'

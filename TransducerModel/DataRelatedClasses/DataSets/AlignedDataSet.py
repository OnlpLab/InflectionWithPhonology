from DataRelatedClasses.DataSets.BaseDataSet import BaseDataSet
from DataRelatedClasses.DataSamples.AlignedDataSample import AlignedDataSample
from defaults import BEGIN_WORD_CHAR, END_WORD_CHAR, BEGIN_WORD, END_WORD
from aligners import smart_align
from typing import List

class AlignedDataSet(BaseDataSet):
    # this dataset aligns its inputs
    def __init__(self, aligner=smart_align, **kwargs):
        super(AlignedDataSet, self).__init__(**kwargs)

        self.aligner = aligner
        use_phonology = kwargs['use_phonology']
        if use_phonology:
            def indices2letters(indices: List[int]) -> list:
                return ['~' if i == '~' else self.vocab.char.i2w[i] for i in indices] if use_phonology else None

            def unwrapper(indices: List[int]) -> list:
                assert indices[0] == BEGIN_WORD and indices[-1] == END_WORD, 'Invalid input!'
                return indices[1:-1]
        else:
            indices2letters, unwrapper = None, None

        # wrapping lemma / word with word boundary tags
        if self.tag_wraps == 'both':
            if use_phonology:
                self.wrapper = lambda s: [BEGIN_WORD] + s + [END_WORD]
            else:
                self.wrapper = lambda s: BEGIN_WORD_CHAR + s + END_WORD_CHAR
        elif self.tag_wraps == 'close':
            self.wrapper = lambda s: s + END_WORD_CHAR
        else:
            self.wrapper = lambda s: s

        print(f'Started aligning with {self.aligner} aligner...')
        if use_phonology:
            if kwargs['self_attention']:
                aligned_pairs = self.aligner([(unwrapper(s.phonemes), s.word_phonological) for s in self.samples], **kwargs)
            else:
                aligned_pairs = self.aligner([(unwrapper(s.lemma), s.word_phonological) for s in self.samples], **kwargs)
            # aligned_pairs[0][0] = [5, 6, 7, 8, 9, 10, 11, 8, ..., 8, 15, 16, 7, 8, 5, 6, 17, '~', '~', ...]
            # aligned_pairs[0][1] = [5, 6, 7, 8, 9, 10, 11, 8, ..., 8, 15, 16, 7, 8, 5, 6, 7, 8, 18, ..., 17]
        else:
            aligned_pairs = self.aligner([(s.lemma_str, s.word_str) for s in self.samples], **kwargs)
            # aligned_pairs[0][0] = 'დაგკარგავთ~~'
            # aligned_pairs[0][1] = 'დაგკარგავდით'
        print('Finished aligning.')

        print('Started building oracle actions...')
        for (al, aw), sample in zip(aligned_pairs, self.samples):
            al, aw = self.wrapper(al), self.wrapper(aw)
            # if use_phon:  al = '<დაგკარგავთ~~>', aw = '<დაგკარგავდით>'
            # else: al = [1, 5, 6, 7, 8, 9, ..., 17, '~', '~', 2], aw = [1, '~', 5, 6, 7, 8, 9, ..., 6, 17, 2]
            if use_phonology: al, aw = indices2letters(al), indices2letters(aw)
            # now al = ['<', '2', '10', '19', '$', '22', .. , '21', '~', '~', '>], aw = ['<', '~', '2', '10', '19', '$', '22', ..., '10', '21', '>']
            self._build_oracle_actions(al, aw, sample=sample, **kwargs)
        print('Finished building oracle actions.')
        print(f'Number of actions: {len(self.vocab.act)}')
        try:
            print(f'Action set: {" ".join(sorted(self.vocab.act.keys()))}')
        except UnicodeError:
            print(f'Action set: {" ".join(sorted(self.vocab.act.keys()))}'.encode('ascii', 'ignore'))

        if self.verbose:
            print('Examples of oracle actions:')
            for a in (s.act_repr for s in self.samples[:20]):
                print(a)  # .encode('utf8')

    def _build_oracle_actions(self, al_lemma, al_word, sample, **kwargs):
        pass

    @classmethod
    def from_file(cls, filename, vocab, **kwargs):
        return super(AlignedDataSet, cls).from_file(filename, vocab, AlignedDataSample, **kwargs)

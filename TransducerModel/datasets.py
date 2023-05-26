import os
import codecs
from random import shuffle

from aligners import smart_align, dumb_align
from defaults import (ALIGN_SYMBOL, STEP, COPY, DELETE,
                      BEGIN_WORD_CHAR, END_WORD_CHAR,
                      BEGIN_WORD, END_WORD, DELETE_CHAR, COPY_CHAR,
                      UNK, SPECIAL_CHARS)
from vocabulary import EditVocab, MinimalVocab

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
        return ''.join(stringified_actions)

def feats2string(pos, feats, vocab):
    if pos:
        pos_str = vocab.pos.i2w[pos] + ';'
    else:
        pos_str = ''
    return  pos_str + ';'.join(vocab.feat.i2w[f] for f in feats)


#############################################################
# DATASETS
#############################################################

class BaseDataSample(object):
    # data sample with encoded features
    def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, out_pos, out_feats, out_feat_str, tag_wraps, vocab):
        self.vocab = vocab          # vocab of unicode strings
        self.lemma = lemma          # list of encoded lemma characters
        self.lemma_str = lemma_str  # unicode string
        self.word = word            # encoded word
        self.word_str = word_str    # unicode string
        self.in_pos = in_pos              # encoded input pos feature
        self.in_feats = in_feats          # set of encoded input features
        self.out_pos = out_pos            # encoded output pos feature
        self.out_feats = out_feats        # set of encoded output features
        self.in_feat_str = in_feat_str    # original serialization of features, unicode
        self.out_feat_str = out_feat_str  # original serialization of features, unicode
        # new serialization of features, unicode
        self.in_feat_repr = feats2string(self.in_pos, self.in_feats, self.vocab)
        self.out_feat_repr = feats2string(self.out_pos, self.out_feats, self.vocab)
        self.tag_wraps = tag_wraps  # were lemma / word wrapped with word boundary tags '<' and '>'?
        
    def __repr__(self):
        return 'Input: {},Features: {}, Output: {}, Features: {}, Wraps: {}'.format(self.lemma_str,
            self.in_feat_repr, self.word_str, self.out_feat_repr, self.tag_wraps)

    @classmethod
    def from_row(cls, vocab, training_data, tag_wraps, verbose, row, sigm2017format=True,
                 no_feat_format=False, pos_emb=True, avm_feat_format=False):
        if sigm2017format:
            input_str, in_feats_str, output_str, out_feats_str = row
            feats_delimiter = ';'
            input_str = remove_pipe(input_str)
            output_str = remove_pipe(output_str)
        else:
            in_feats_str, input_str, out_feats_str, output_str = row
            feats_delimiter = ';'
            # feats_delimiter = u','
        # encode features as integers
        # POS feature treated separately
        assert isinstance(input_str, str), input_str
        assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
        assert isinstance(output_str, str), output_str
        assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)
        assert isinstance(in_feats_str, str), in_feats_str
        assert isinstance(out_feats_str, str), out_feats_str
        # `avm_feat_format=True` implies that `pos_emb=False`
        if avm_feat_format: assert not pos_emb
        # encode input characters
        input = [vocab.char[c] for c in input_str] #.split()] # todo oracle
        # encode word
        word = vocab.word[output_str] #.replace(' ','')] # todo oracle
        in_feats = in_feats_str.split(feats_delimiter) if not no_feat_format else ['']
        out_feats = out_feats_str.split(feats_delimiter) if not no_feat_format else ['']
        if pos_emb:
            # encode features and, separately, pos
            in_pos = vocab.pos[in_feats[0]]
            out_pos = vocab.pos[out_feats[0]]
            in_feats = [vocab.feat[f] for f in set(in_feats[1:])]
            out_feats = [vocab.feat[f] for f in set(out_feats[1:])]
        else:
            in_pos, out_pos  = None, None
            if avm_feat_format:
                # map from encoded feature names to encoded features
                in_feats = {vocab.feat_type[f.split('=')[0]] : vocab.feat[f] for f in set(in_feats)}
                out_feats = {vocab.feat_type[f.split('=')[0]] : vocab.feat[f] for f in set(out_feats)}
            else:
                in_feats = [vocab.feat[f] for f in set(in_feats)]
                out_feats = [vocab.feat[f] for f in set(out_feats)]
        # wrap encoded input with (encoded) boundary tags
        if tag_wraps == 'both':
            input = [BEGIN_WORD] + input + [END_WORD]
        elif tag_wraps == 'close':
            input = input + [END_WORD]
        # print features and input at a high verbosity level
        if verbose == 2:
            # print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, output_str, input_str, pos, feats)
            print('POS & features from {}, {}: {}, {} --> {}, {}'.format(input_str, output_str, in_pos, in_feats, out_pos, out_feats))
            print('input encoding: {}'.format(input))
        return cls(input, input_str, in_pos, in_feats, in_feats_str, word, output_str, out_pos, out_feats, out_feats_str, tag_wraps, vocab)


class PCFPDataSample(object):
    # def __init__(self, lemma, lemma_str, in_pos, in_feats, in_feat_str, word, word_str, out_pos, out_feats, out_feat_str, tag_wraps, vocab):
    def __init__(self, inputs,                                            word, word_str, out_pos, out_feats, out_feat_str, tag_wraps, vocab):
        self.vocab = vocab          # vocab of unicode strings
        self.word = word            # encoded word
        self.word_str = word_str    # unicode string
        self.out_pos = out_pos            # encoded output pos feature
        self.out_feats = out_feats        # set of encoded output features
        self.out_feat_str = out_feat_str  # original serialization of features, unicode
        # new serialization of features, unicode
        self.in_feat_repr = feats2string(inputs[0][2], inputs[0][3], self.vocab)
        self.out_feat_repr = feats2string(self.out_pos, self.out_feats, self.vocab)
        self.tag_wraps = tag_wraps  # were lemma / word wrapped with word boundary tags '<' and '>'?
        self.samples = [BaseDataSample(*sample, word=word, word_str=word_str, out_pos=out_pos, out_feats=out_feats,
                                       out_feat_str=out_feat_str, tag_wraps=tag_wraps, vocab=vocab)
                                       for sample in inputs]

    @classmethod
    def from_row(cls, vocab, training_data, tag_wraps, verbose, list_of_rows, sigm2017format=True,
                 no_feat_format=False, pos_emb=True, avm_feat_format=False):
        '''
        list of rows: List(input, input_feats, output, output_feats)
        '''
        feats_delimiter = ','

        # treat output first. it's the same for all rows
        output_str, out_feats_str = list_of_rows[0][-2:]
        assert isinstance(output_str, str), output_str
        assert not any(c in output_str for c in SPECIAL_CHARS), (output_str, SPECIAL_CHARS)
        assert isinstance(out_feats_str, str), out_feats_str
        if avm_feat_format: assert not pos_emb

        word = vocab.word[output_str]
        out_feats = out_feats_str.split(feats_delimiter) if not no_feat_format else ['']

        if pos_emb:
            out_pos = vocab.pos[out_feats[0]]
            out_feats = [vocab.feat[f] for f in set(out_feats[1:])]
        else:
            out_pos = None
            if avm_feat_format:
                # map from encoded feature names to encoded features
                out_feats = {vocab.feat_type[f.split('=')[0]] : vocab.feat[f] for f in set(out_feats)}
            else:
                out_feats = [vocab.feat[f] for f in set(out_feats)]

        # now treat the multiple inputs
        list_of_inputs = []
        for row in list_of_rows:
            input_str, in_feats_str = row[:2]
            assert isinstance(input_str, str), input_str
            assert not any(c in input_str for c in SPECIAL_CHARS), (input_str, SPECIAL_CHARS)
            assert isinstance(in_feats_str, str), in_feats_str

            input = [vocab.char[c] for c in input_str]
            in_feats = in_feats_str.split(feats_delimiter) if not no_feat_format else ['']

            if pos_emb:
                # encode features and, separately, pos
                in_pos = vocab.pos[in_feats[0]]
                in_feats = [vocab.feat[f] for f in set(in_feats[1:])]
            else:
                in_pos = None
                if avm_feat_format:
                    # map from encoded feature names to encoded features
                    in_feats = {vocab.feat_type[f.split('=')[0]]: vocab.feat[f] for f in set(in_feats)}
                else:
                    in_feats = [vocab.feat[f] for f in set(in_feats)]

            if tag_wraps == 'both':
                input = [BEGIN_WORD] + input + [END_WORD]
            elif tag_wraps == 'close':
                input = input + [END_WORD]

            list_of_inputs.append([input, input_str, in_pos, in_feats, in_feats_str])

            if verbose == 2:
                # print u'POS & features from {}, {}, {}: {}, {}'.format(feat_str, output_str, input_str, pos, feats)
                print('POS & features from {}, {}: {}, {} --> {}, {}'.format(input_str, output_str, in_pos, in_feats,
                                                                        out_pos, out_feats))
                print('input encoding: {}'.format(input))

        return cls(list_of_inputs, word, output_str, out_pos, out_feats, out_feats_str, tag_wraps, vocab)


class AlignedDataSample(BaseDataSample):
    # data sample with encoded oracle actions derived from character alignment of lemma and word
    def set_actions(self, actions, aligned_lemma, aligned_word):
        self.actions = actions              # list of indices
        # serialization of actions as unicode string
        self.act_repr = action2string(self.actions, self.vocab)
        self.aligned_lemma = aligned_lemma  # unicode string
        self.aligned_word = aligned_word    # unicode string

    def __repr__(self):
        return 'Input: {},Features: {}, Output: {}, Features: {}, Actions: {}'.format(
            self.lemma_str, self.in_feat_repr, self.word_str, self.out_feat_repr, self.act_repr)


class PhonlogicalDataSample(BaseDataSample):
    pass


class BaseDataSet(object):
    # class to hold an encoded dataset
    def __init__(self, filename, samples, vocab, training_data, tag_wraps, verbose, **kwargs):
        self.filename = filename
        self.samples = samples
        self.vocab = vocab
        self.length = len(self.samples)
        self.training_data = training_data
        self.tag_wraps = tag_wraps
        self.verbose = verbose
    
    def __len__(self): return self.length

    @classmethod
    def from_file(cls, filename, vocab, DataSample=BaseDataSample,
                  encoding='utf8', delimiter='\t', sigm2017format=True, no_feat_format=False,
                  pos_emb=True, avm_feat_format=False, tag_wraps='both', verbose=True, **kwargs):
        # filename (str):   tab-separated file containing morphology reinflection data:
        #                   lemma word feat1;feat2;feat3...
        if isinstance(filename, list):
            filename, hallname = filename
            print('adding hallucinated data from', hallname)
        else:
            hallname = None

        training_data = True if 'inflec_data' in os.path.basename(filename) or 'all_ns' in os.path.basename(filename) \
                                or 'train' in os.path.basename(filename) else False
        if training_data:
            print('=====TRAIN TRAIN TRAIN=====')
        else:
            print('=====TEST TEST TEST=====')
        print(filename)

        # training_data = True if 'train' in os.path.basename(filename) else False
        datasamples = []
        
        print('Loading data from file: {}'.format(filename))
        print('These are {} data.'.format('training' if training_data else 'holdout'))
        print('Word boundary tags?', tag_wraps)
        print('Verbose?', verbose)

        if avm_feat_format:
            # check that `avm_feat_format` and `pos_emb` does not clash
            if pos_emb:
                print('Attribute-value feature matrix implies that no specialized pos embedding is used.')
                pos_emb = False
        
        with codecs.open(filename, encoding=encoding) as f:
            for row in f:
                split_row = row.strip().split(delimiter)
                sample = DataSample.from_row(vocab, training_data, tag_wraps, verbose,
                                         split_row, sigm2017format, no_feat_format,
                                         pos_emb, avm_feat_format)
                datasamples.append(sample)

        if hallname:
            old_len = len(datasamples)
            if len(datasamples)>5000:
                shuffle(datasamples)
                datasamples = datasamples[:5000]
            with codecs.open(hallname, encoding=encoding) as f:
                for row in f:
                    split_row = row.strip().split(delimiter)
                    letters = set(split_row[0]) | set(split_row[3])
                    if '|' in letters:
                        continue
                    sample = DataSample.from_row(vocab, training_data, tag_wraps, verbose,
                                                 split_row, sigm2017format, no_feat_format,
                                                 pos_emb, avm_feat_format)
                    datasamples.append(sample)
            print('hallucinated data added. training expanded from {} to {} examples'.format(old_len, len(datasamples)))

        return cls(filename=filename, samples=datasamples, vocab=vocab,
                   training_data=training_data, tag_wraps=tag_wraps, verbose=verbose, **kwargs)

class PCFPDataSet(BaseDataSet):
    @classmethod
    def from_file(cls, filename, vocab, DataSample=PCFPDataSample,
                  encoding='utf8', delimiter='\t', sigm2017format=True, no_feat_format=False,
                  pos_emb=True, avm_feat_format=False, tag_wraps='both', verbose=True, **kwargs):
        print('Loading data from file: {}'.format(filename))
        print('===========ATTENTION============')
        print('These are holdout data in PCFP format.')
        print('Word boundary tags?', tag_wraps)
        print('Verbose?', verbose)

        assert not avm_feat_format

        answers_filename = filename
        covered_filename = os.path.splitext(filename)[0]+ '.3.txt'

        samples = []
        inputs = []
        outputs = []
        with codecs.open(answers_filename, encoding=encoding) as answers, \
                codecs.open(covered_filename, encoding=encoding) as covered:
            for ans_line in answers:
                cov_line = covered.readline()
                if not ans_line.strip():
                    for ans, ans_feats in outputs:
                        for_sample = []
                        for inp, inp_feats in inputs:
                            for_sample.append([inp, inp_feats, ans, ans_feats])
                        if not for_sample:
                            continue
                        sample = DataSample.from_row(vocab, False, tag_wraps, verbose,
                                             for_sample, sigm2017format, no_feat_format,
                                             pos_emb, avm_feat_format)
                        samples.append(sample)

                    inputs = []
                    outputs = []
                    continue

                ans_line = ans_line.strip().split('\t')
                cov_line = cov_line.strip().split('\t')
                if len(cov_line) == 2:
                    inputs.append(cov_line)
                else:
                    outputs.append(ans_line)

        return cls(filename=filename, samples=samples, vocab=vocab,
                   training_data=False, tag_wraps=tag_wraps, verbose=verbose, **kwargs)


class AlignedDataSet(BaseDataSet):
    # this dataset aligns its inputs
    def __init__(self, aligner=smart_align, **kwargs):
        super(AlignedDataSet, self).__init__(**kwargs)

        self.aligner = aligner
        # wrapping lemma / word with word boundary tags
        if self.tag_wraps == 'both':
            self.wrapper = lambda s: BEGIN_WORD_CHAR + s + END_WORD_CHAR # todo oracle [BEGIN_WORD_CHAR] [END_WORD_CHAR]
        elif self.tag_wraps == 'close':
            self.wrapper = lambda s: s + END_WORD_CHAR # todo oracle [END_WORD_CHAR]
        else:
            self.wrapper = lambda s: s
        
        print('Started aligning with {} aligner...'.format(self.aligner))
        aligned_pairs = self.aligner([(s.lemma_str, s.word_str) for s in self.samples], **kwargs)
        print('Finished aligning.')

        print('Started building oracle actions...')
        for (al, aw), s in zip(aligned_pairs, self.samples):
            al = self.wrapper(al)
            aw = self.wrapper(aw)
            self._build_oracle_actions(al, aw, sample=s, **kwargs)
        print('Finished building oracle actions.')
        print('Number of actions: {}'.format(len(self.vocab.act)))
        print('Action set: {}'.format(' '.join(sorted(self.vocab.act.keys()))))
        
        if self.verbose:
            print('Examples of oracle actions:')
            for a in (s.act_repr for s in self.samples[:20]):
                print(a) #.encode('utf8')

    def _build_oracle_actions(self, al_lemma, al_word, sample, **kwargs):
        pass
    
    @classmethod
    def from_file(cls, filename, vocab, **kwargs):
        return super(AlignedDataSet, cls).from_file(filename, vocab, AlignedDataSample, **kwargs)

    
class MinimalDataSet(AlignedDataSet):
    # this dataset builds actions with
    # Algorithm of Aharoni & Goldberg 2017
    def _build_oracle_actions(self, lemma, word, sample, **kwargs):
        # Aharoni & Goldberg 2017 Algorithm 1
        actions = []
        alignment_len = len(lemma)
        for i, (l, w) in enumerate(zip(lemma, word)):
            if w == ALIGN_SYMBOL:
                actions.append(STEP)
            else:
                actions.append(self.vocab.act[w])  # encode w
                if i+1 < alignment_len and lemma[i+1] != ALIGN_SYMBOL:
                    actions.append(STEP)
        if self.verbose == 2:
            print('{}\n{}\n{}\n'.format(word,
                action2string(actions, self.vocab), lemma))

        sample.set_actions(actions, lemma, word)

    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, MinimalVocab)
        else:
            vocab = MinimalVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format,
                                 param_tying=param_tying)
        print(vocab)
        return super(MinimalDataSet, cls).from_file(filename, vocab, pos_emb=pos_emb,
                                                    avm_feat_format=avm_feat_format, **kwargs)


class EditDataSet(AlignedDataSet):
    # this dataset uses COPY action
    def __init__(self, try_reverse=False, substitution=False, copy_as_substitution=False,
                 reorder_deletes=True, freq_check=(0.1, 0.3), **kwargs):
        # "try reverse" only makes sense with dumb aligner
        self.try_reverse = try_reverse and self.aligner == dumb_align  # @TODO Fix bug
        if self.try_reverse:
            print('USING STRING REVERSING WITH DUMB ALIGNMENT...')
            print('USING DEFAULT ALIGN SYMBOL ~')
        self.copy_as_substitution = copy_as_substitution
        self.substitution = substitution
        if copy_as_substitution is True:
            self.substitution = True
            print('TREATING COPY AS SUBSTITUTIONS')
        if self.substitution is True:
            self.reorder_deletes = False
            print('USING SUBSTITUTION ACTIONS, NOT REORDERING DELETES')
        else:
            self.reorder_deletes = reorder_deletes
        # "frequency check" for COPY and DELETE actions
        self.freq_check = freq_check
            
        super(EditDataSet, self).__init__(**kwargs)
        if self.freq_check:
            copy_low, delete_high = self.freq_check
            # some stats on actions
            action_counter = self.vocab.act.freq()
            #print action_counter.values()
            freq_delete = action_counter[DELETE] / sum(action_counter.values())
            freq_copy = action_counter[COPY] / sum(action_counter.values())
            print(('Alignment results: COPY action freq {:.3f}, '
                   'DELETE action freq {:.3f}'.format(freq_copy, freq_delete)))
            if freq_copy < copy_low:
                print('WARNING: Too few COPY actions!\n')
            if freq_delete > delete_high:
                print('WARNING: Many DELETE actions!\n')

    def _build_oracle_actions(self, lemma, word, sample, **kwargs):
        # Makarov et al 2017 Algorithm 1
        def _build(lemma, word):
            actions = []
            alignment_len = len(lemma)
            has_copy = False
            for i, (l, w) in enumerate(zip(lemma, word)):
                if l == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[w])
                elif w == ALIGN_SYMBOL:
                    actions.append(self.vocab.act[DELETE_CHAR])
                elif l == w:
                    if i+1 == alignment_len:
                        # end of string => insert </s>
                        actions.append(self.vocab.act[w])
                    elif self.copy_as_substitution:
                        # treat copy as another substitution action
                        actions.append(self.vocab.act[w+'@'])
                    else:
                        # treat copy as a special action
                        actions.append(self.vocab.act[COPY_CHAR])
                        has_copy = True
                else:
                    # substitution
                    if self.substitution:
                        subt = self.vocab.act[w+'@'],
                        #subt = (self.vocab.act[u'@' + l + w + u'@'],)
                    else:
                        subt = self.vocab.act[DELETE_CHAR], self.vocab.act[w]
                    actions.extend(subt)
            return actions, has_copy
        
        actions, has_copy = _build(lemma, word)
        
        if self.try_reverse and has_copy:
            # no copying is being done, probably
            # this sample uses prefixation. Try aligning
            # original pair from the end:
            reversed_pair = sample.lemma[::-1], sample.word[::-1]
            [(new_al_lemma, new_al_word)] = self.aligner([reversed_pair], ALIGN_SYMBOL)
            ractions, has_copy = _build(new_al_lemma[::-1], new_al_word[::-1])
            if has_copy:
                print(('Reversed aligned: {} => {}\n'
                       'Forward alignment: {}, REVERSED alignment: {}'.format(
                        al_lemma, al_word,
                        action2string(actions,  self.vocab),
                        action2string(ractions, self.vocab))))
                actions = ractions
                       
        if self.reorder_deletes:
            reordered_actions = []
            suffix = []
            for i, c in enumerate(actions):
                if i == 0 or c == COPY:
                    reordered_actions.append(c)
                    # count deletes and store inserts
                    # between two copy actions
                    inserts = []
                    deletes = 0
                    for b in actions[i+1:]:
                        if b == COPY:
                            # copy
                            break
                        elif b == DELETE:
                            # delete
                            deletes += 1
                        else:
                            inserts.append(b)
                    between_copies = [DELETE]*deletes + inserts
                    reordered_actions.extend(between_copies)
            actions = reordered_actions + suffix

        if self.verbose == 2:
            print('{}\n{}\n{}\n'.format(word,
                                         action2string(actions, self.vocab),
                                         lemma))

        sample.set_actions(actions, lemma, word)

    @classmethod
    def from_file(cls, filename, vocab=None, pos_emb=True, avm_feat_format=False,
                  param_tying=False, **kwargs):
        if vocab:
            assert isinstance(vocab, EditVocab)
        else:
            vocab = EditVocab(pos_emb=pos_emb, avm_feat_format=avm_feat_format,
                              param_tying=param_tying)
        print(vocab)
        return super(EditDataSet, cls).from_file(filename, vocab, pos_emb=pos_emb,
                                                 avm_feat_format=avm_feat_format, **kwargs)





if __name__ == "__main__":
    import os
    from defaults import DATA_PATH

    fn = os.path.join(DATA_PATH, 'russian-train-low')
    ds = MinimalDataSet.from_file(fn, verbose=2,
                                  tag_wraps='both',
                                  iterations=5)
    vocab = ds.vocab
    print()
    fn = os.path.join(DATA_PATH, 'russian-dev')
    ds = MinimalDataSet.from_file(fn, vocab=vocab,
                                  verbose=True,
                                  tag_wraps='both',
                                  iterations=5)

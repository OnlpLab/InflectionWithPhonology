from typing import List

import dynet as dy
from _dynet import Expression

from Models.self_attention import MultiHeadAttentionLayer

#############################################################
# Stack RNNs and biRNNs
#############################################################

# from Chris Dyer and Co.'s EMNLP 2016 tutorial:
class StackRNN(object):
    def __init__(self, rnn, p_empty_embedding=None):
        self.s = [(rnn.initial_state(), None)]
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)

    def push(self, expr, extra=None):
        self.s.append((self.s[-1][0].add_input(expr), extra))

    def pop(self):
        return self.s.pop()[1]  # return "extra" (i.e., whatever the caller wants or None)

    def embedding(self):
        # work around since inital_state.output() is None
        return self.s[-1][0].output() if len(self.s) > 1 else self.empty

    def __len__(self):
        return len(self.s) - 1


class DeleteRNN(StackRNN):
    def clear_all(self):
        self.s = self.s[:1]


class StackBiRNN(object):
    def __init__(self, frnn, brnn, p_empty_embedding=None):
        self.frnn = frnn
        self.brnn = brnn
        self.empty = None
        if p_empty_embedding:
            self.empty = dy.parameter(p_empty_embedding)

    def transduce(self, embs, extras=None):
        fs = self.frnn.initial_state()
        bs = self.brnn.initial_state()
        fs_states = fs.add_inputs(embs)  # 1, 2, 3, 4
        bs_states = reversed(bs.add_inputs(reversed(embs)))  # 1, 2, 3, 4
        self.s = [(fs, bs, None)] + reversed(list(zip(fs_states, bs_states, extras)))  # 0, 4, 3, 2, 1

    def pop(self):
        return self.s.pop()[-1]  # return "extra" (i.e., whatever the caller wants or None)

    def embedding(self):
        if len(self.s) > 1:
            fs = self.s[-1][0].output()
            bs = self.s[-1][1].output()
            emb = dy.concatenate([fs, bs])
        else:
            # work around since inital_state.output() is None
            emb = self.empty
        return emb

    def __len__(self):
        return len(self.s) - 1


class Encoder(object):
    def __init__(self, frnn, brnn, self_attention_layer: MultiHeadAttentionLayer = None, max_phoneme_size = None):
        self.forward_rnn = frnn
        self.backward_rnn = brnn
        self.self_attention_layer = self_attention_layer
        self.max_phoneme_size = max_phoneme_size # mostly 3 or 4

        self.s = None

    def apply_self_attention(self, embeddings, phonemes_embeddings):
        sos_and_eos_embs = [embeddings[0], embeddings[-1]] # same as [phonemes_embeddings[0], phonemes_embeddings[-1]]
        embeddings, phonemes_embeddings = embeddings[1:-1], phonemes_embeddings[1:-1] # ignore sos and eos
        assert (len(embeddings) / len(phonemes_embeddings)).is_integer()

        attended_phonemes = []
        for i, p_emb in enumerate(phonemes_embeddings):
            features_embs = embeddings[self.max_phoneme_size * i: self.max_phoneme_size * (i + 1)]
            features_embs = dy.concatenate(features_embs, d=1)
            attended_phoneme = self.self_attention_layer(p_emb, features_embs, None)
            attended_phonemes.append(attended_phoneme)

        assert len(attended_phonemes) == len(phonemes_embeddings)
        attended_phonemes = [sos_and_eos_embs[0], *attended_phonemes, sos_and_eos_embs[1]]

        return attended_phonemes

    def transduce(self, embeddings: List[Expression], extras: List[int] = None, phonemes_embeddings=None,
                  phonemes_extras: List[int] = None):
        """
        :param embeddings: a list of embedding vectors (of type Expression)
        :param extras: a list of the lemma vocab.char values, or None
        :param phonemes_embeddings: a list of the lemma's phonemes characters (of type Expression), or None
        :param phonemes_extras: a list of the lemma's phonemes vocab.char values, or None
        """
        assert bool(phonemes_extras) == bool(phonemes_embeddings) == bool(self.self_attention_layer)

        if phonemes_extras:
            embeddings = self.apply_self_attention(embeddings, phonemes_embeddings)
            # now embeddings.shape == (m, CHAR_DIM), where m = len(w2p(lemma, 'phonemes'))

        fs = self.forward_rnn.initial_state()
        bs = self.backward_rnn.initial_state()
        forward_states = fs.add_inputs(embeddings)  # 1, 2, 3, 4
        backward_states = reversed(bs.add_inputs(reversed(embeddings)))  # 1, 2, 3, 4

        if phonemes_extras:
            self.s = list(reversed(list(zip(forward_states, backward_states, phonemes_extras))))  # 4, 3, 2, 1
        else:
            self.s = list(reversed(list(zip(forward_states, backward_states, extras))))  # 4, 3, 2, 1

        # special treatment for the final element
        final_s = self.s[0]
        self.final_embedding = dy.concatenate([final_s[0].output(), final_s[1].output()])
        self.final_extra = final_s[2]

    def embedding(self, extra=False):
        if len(self.s) > 1:
            fs, bs, e = self.s[-1]
            output = dy.concatenate([fs.output(), bs.output()])
        else:
            e = self.final_extra
            output = self.final_embedding
        if extra:
            output = output, e
        return output

    def pop(self):
        return self.s.pop()[-1]  # return "extra" (i.e., whatever the caller wants or None)

    def __len__(self):
        return len(self.s)

    def copy(self):
        # Not used in the basic Transducer
        encoder = Encoder(self.forward_rnn, self.backward_rnn)
        encoder.s = list(self.s)  # copy
        encoder.final_embedding = self.final_embedding
        encoder.final_extra = self.final_extra
        return encoder

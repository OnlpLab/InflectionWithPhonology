import random
import torch.nn as nn

import hyper_params_config as hp
import utils
from PhonologyConverter.languages_setup import MAX_FEAT_SIZE
from utils import get_abs_offsets, postprocessBatch, torch

torch.manual_seed(hp.SEED)

def print_readable_tensor(x): print([utils.srcField.vocab.itos[i] for i in x])  # used for debugging purposes

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=utils.srcField.vocab.stoi['<pad>'])

        self.input_size = input_size
        self.embedding_size = embedding_size
        if hp.PHON_USE_ATTENTION:
            self.phon_selfAttention = nn.MultiheadAttention(self.embedding_size, num_heads=2)
            # https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html - a useful explanation of self-attention
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        # x: (seq_length, N) where N is batch size

        embedding = self.embedding(x)
        if hp.PHON_UPGRADED:
            phon_delim, pad_idx = [utils.srcField.vocab.stoi[e] for e in ['$', '<pad>']]
            iter_embeds = embedding.permute(1, 0, 2)
            new_embs = []

            for i, seq in enumerate(x.t()):  # iterating over the batch samples
                mat = iter_embeds[i]
                abs_offsets = get_abs_offsets(seq, phon_delim,
                                              phon_max_len=MAX_FEAT_SIZE + 1 if hp.PHON_USE_ATTENTION else MAX_FEAT_SIZE)
                if hp.PHON_USE_ATTENTION:
                    res_vecs, vecs_indices = [], torch.cat([torch.arange(o, o + MAX_FEAT_SIZE + 1) for o in abs_offsets], dim=0)
                    phon_vecs = mat[vecs_indices]  # get all the phonological vector-triplets and assign them to phon_vecs
                    vec_triplets = torch.split(phon_vecs, MAX_FEAT_SIZE + 1, dim=0)  # splitting to a tuple of triplets

                    for triplet in vec_triplets:  # iterating over all the phonemes of the sample. Couldn't parallelize this part
                        triplet = triplet.unsqueeze(1)  # shape = [MAX_FEAT + 1, 1, embed_size]

                        # Now, use the last vector (E['t͡sʰ']) as the query of the self-attention
                        features, phoneme = triplet[:-1], triplet[-1, None]  # shapes = [MAX_FEAT, 1, embed_size], [1, 1, embed_size] ([.., None] <=> unsqueeze)

                        t, _ = self.phon_selfAttention(query=phoneme, key=features,
                                                       value=features)  # shape = [1, 1, embed_size] (same as query)
                        t = t.squeeze(1)
                        # Finally, we need to reduce the sequence length to the new one, where every phoneme is
                        # represented by the single vector t. In practice, it's done more efficiently later inside postprocessBatch.

                        # t_with_zeros = torch.zeros_like(triplet).squeeze(1)
                        # t_with_zeros[0] = t*(MAX_FEAT_SIZE+1)
                        # new_t = t_with_zeros # the 3 last lines are equivalent to the next one ( Mean(x, x, x, x) = 4 * Mean(x, 0, 0, 0) = x )
                        new_t = t.repeat(MAX_FEAT_SIZE + 1, 1)
                        res_vecs.append(new_t)
                    mat.index_put_((vecs_indices,), torch.cat(res_vecs, dim=0))

                new_mat = postprocessBatch(mat, abs_offsets)
                new_embs.append(new_mat)

            lens = [e.shape[0] for e in new_embs]
            if len(set(lens)) > 1:  # if all updated sequences have same lengths, skip the following part.
                # Find the maximal new length, and pad the new batch with embeddings of <pad>
                max_len = max(lens)
                with torch.no_grad():
                    pad_token_embedding = self.embedding(torch.tensor([[pad_idx]], device=utils.device)).squeeze(0)
                for i, new_mat in enumerate(new_embs):
                    if new_mat.shape[0] < max_len:
                        new_embs[i] = torch.cat((new_mat, pad_token_embedding.repeat((max_len - new_mat.shape[0]), 1)))
            embedding = torch.stack(new_embs, dim=1)

        # Important change: instead of applying Dropout on embedding here, we apply it on the hidden vectors after the FC layers.
        # embedding = self.dropout(embedding)
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.dropout(self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2)))
        cell = self.dropout(self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2)))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(
            self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell, return_attn=False):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        # Not applying Dropout on the embeddings!
        # embedding = self.dropout(self.embedding(x))
        embedding = self.embedding(x)
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)
        # Similarly to the Encoder, here as well we apply Dropout on the hidden vectors.
        hidden, cell = self.dropout(hidden), self.dropout(cell)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size) # actually (N, output_size)?

        attn = attention if return_attn else None
        return predictions, hidden, cell, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(utils.trgField.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(utils.device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell, _ = self.decoder(x, encoder_states, hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

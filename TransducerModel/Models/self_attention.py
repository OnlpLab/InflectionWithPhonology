# Basic classes, such as MultiHeadAttention
import numpy as np
from dynet import affine_transform, cmult, concatenate, dropout_dim, reshape, softmax, transpose


def make_time_distributed(x):
    """
    Flatten multi-dim matrix to single batched vector for efficient calculation
    Based on work by Cong Duy Vu Hoang
    """
    d = x.dim()
    b = d[1]
    d = d[0]

    total_words = d[1] * b
    return reshape(x, (d[0], 1), total_words)

def make_reverse_time_distributed(x, seq_len, b_):
    d = x.dim()[0]
    return reshape(x, (d[0], seq_len), b_)


class LinearLayer(object):
    """
    Simple Linear Layer (w/ or w/o bias)
    
    Class which employs a basic Linear Layer, with an option to efficiently run on sequences of large batches
    """

    def __init__(self, model, input_dim, output_dim, have_bias, use_he_init=True, name=""):
        self._have_bias = have_bias

        init_w = None
        init_bias = None
        if use_he_init:
            # may not be defined, so just ry
            try:
                init_w = LeCunUniformInitializer(input_dim)
                init_bias = LeCunUniformInitializer(output_dim)
            except NameError:
                pass

        self._p_W = model.add_parameters((output_dim, input_dim), init=init_w, name=name + '.ll.w')
        if have_bias:
            self._p_b = model.add_parameters(output_dim, init=init_bias, name=name + '.ll.b')

    def __call__(self, x, flatten_sequence=False):
        """
        Run the input Expression x through the linear layer
        
        Args:
            x (Expression): The vector to run through the linear layer (can be batched) 
            flatten_sequence (bool): If the expression is 2 dimensions, and we run to all the columns through the linear layer (treat ({x,y},n) as ({x}, y*n) for efficient multiplication)
        """
        x_in = x
        if flatten_sequence:
            x_in = make_time_distributed(x)

        if self._have_bias:
            try:
                x_out = affine_transform([self._p_b, self._p_W, x_in])
            except TypeError: # in dynet==2.0.2 _dynet.Parameters and _dynet.Expression aren't castable to each other
                x_out = affine_transform([self._p_b.expr(), self._p_W.expr(), x_in])
        else:
            x_out = self._p_W * x_in

        if flatten_sequence:
            d = x.dim()
            b = d[1]
            d = d[0]
            x_out = make_reverse_time_distributed(x_out, d[1], b)

        return x_out

class MultiHeadAttentionLayer(object):
    """
        Multi-Head Attention Layer for multi-head attention computing (faster)
        
        Currently only support the luong attention type (dot-product)
    """

    def __init__(self, model, dim, nheads, use_bias=False, apply_future_blinding=False, name=''):
        self._v_l_Q = []
        self._v_l_K = []
        self._v_l_V = []
        for i in range(nheads):
            self._v_l_Q.append(LinearLayer(model, dim, dim // nheads, use_bias, name=f'{name}.attn.h{str(i)}.linear-query'))
            self._v_l_K.append(LinearLayer(model, dim, dim // nheads, use_bias, name=f'{name}.attn.h{str(i)}.linear-keys'))
            self._v_l_V.append(LinearLayer(model, dim, dim // nheads, use_bias, name=f'{name}.attn.h{str(i)}.linear-values'))

        # final layer
        self._l_O = LinearLayer(model, dim, dim, use_bias, name=f'{name}.attn.linear-final')

        self._att_scale = 1.0 / np.sqrt(dim / nheads)
        self.dim = dim
        self.nheads = nheads
        self._apply_future_blind_mask = apply_future_blinding

    def __call__(self, query, keys, i_mask, dropout=0.0):
        """
            Calculate the attention weights and apply them to the keys using the dot-product attention type
            query: a list of len seq_len, each item is an Expression of shape (CHAR_DIM, )
        """

        v_atts = []
        for (_l_Q, _l_K, _l_V) in zip(self._v_l_Q, self._v_l_K, self._v_l_V):
            i_Q = _l_Q(query)  # ((dk,lQ), batch)
            i_K = _l_K(keys)  # ((dk,lK), batch)
            i_V = _l_V(keys)  # ((dk,lK), batch)

            # calculate alphas and apply scale
            i_alphas = (transpose(i_K) * i_Q) * self._att_scale
            # apply source mask - give a very low score to items not in keys
            if i_mask is not None:
                i_alphas = i_alphas + i_mask.get_k_mask()

            # apply the tgt mask
            if self._apply_future_blind_mask:
                i_alphas = i_alphas + i_mask.i_mask_fb
                # softmax
            i_alphas = softmax(i_alphas)
            # apply query mask - zero out values not in query
            if i_mask is not None:
                i_alphas = cmult(i_alphas, i_mask.get_q_mask())

            if dropout > 0.0:
                # as before, column-major dropout
                i_alphas = dropout_dim(i_alphas, 1, dropout)

            i_proj = i_V * i_alphas
            v_atts.append(i_proj)  # ((dk, ly), batch_size)

        return self._l_O(concatenate(v_atts))  # ((dim, ly), batch_size)

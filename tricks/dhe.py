# Copyright (c) Chanwoo Park 2024
#


from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sympy


class DHEFFN(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        self.dense = nn.Linear(input_dim, output_dim)
        nn.init.uniform_(self.dense, np.sqrt(1 / k))
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        x = self.dense(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def generate_random_primes(k, m):
    # generate k random primes larger than m
    p_list = []
    for _ in range(k):
        p = m
        while not sympy.isprime(p):
            p = np.random.randint(m, 2 * m)
        p_list.append(p)
    return torch.Tensor(p_list, dtype=torch.long)


def generate_random_numbers(k, m):
    # generate k random numbers in [1, m)
    a_list = np.random.randint(1, m, k)
    b_list = np.random.randint(1, m, k)
    return torch.Tensor(a_list, dtype=torch.long), torch.Tensor(
        b_list, dtype=torch.long
    )


class DHE(nn.Module):
    r"""Computes sums or means over multiple feature embeddings, calculated with hash function and DNN networks.

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=0)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=0)``,

    Known Issues:

    # TODO: update this chanwoo. arguments are not updated
    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    # TODO: update this chanwoo
    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.


    Output shape: `(B, embedding_dim)`

    """

    def __init__(
        self,
        a_list,
        b_list,
        p_list,
        embedding_dim=32,
        k=1024,  # number of hash function
        h=5,  # number of hidden layers
        d_nn=1800,  # dimension of hidden layers
        m=1000000,
        uniform=False,
        reduce="sum",
    ):
        super(DHE, self).__init__()

        assert reduce in ["mean", "add"], "Not valid reduce!"
        self.k = k
        self.h = h
        self.d_nn = d_nn
        self.m = m
        self.reduce = reduce

        self.a_list = a_list
        self.b_list = b_list
        self.p_list = p_list

        self.layers = nn.ModuleList()
        # first layer
        # in: k, out: d_nn
        self.layers.append(DHEFFN(k, d_nn, k))
        # hidden layers
        for _ in range(1, h - 1):
            self.layers.append(DHEFFN(d_nn, d_nn, k))
        # last layer
        # in: d_nn, out: embedding_dim
        self.layers.append(DHEFFN(d_nn, embedding_dim, k))
        self.uniform = uniform

    def _transform(self, x):
        # input: s x 1
        # output: s x k
        x = x.unsqueeze(-1).expand(-1, len(self.a_list))
        # transform: x[b,i] = (a_list[i] * x[i] + b_list[i]) % p_list[i] % m
        x = (self.a_list * x + self.b_list) % self.p_list % self.m + 1
        x = (x - 1) / (self.m - 1) * 2 - 1
        e = torch.zeros_like(x)
        x_even = x[:, 0::2]
        x_odd = x[:, 1::2]
        e[:, 0::2] = torch.sqrt(-2 * torch.log(x_even)) * torch.cos(2 * np.pi * x_odd)
        e[:, 1::2] = torch.sqrt(-2 * torch.log(x_even)) * torch.sin(2 * np.pi * x_odd)

        # reduce operation
        # e[s, k] -> e[k]
        if self.reduce == "sum":
            e = e.sum(dim=0)
        elif self.reduce == "mean":
            e = e.mean(dim=0)
        return e

    def _forward(self, x):
        t = self._transform(x, self.a_list, self.b_list, self.p_list, self.m)
        for layer in self.layers:
            t = layer(t)
        return t

    def forward(self, input, offsets):
        # input: Sequence of indices into the embedding tables
        # offsets:  It specifies the starting index position of each bag (sequence) in input.
        # use offset to separate inputs

        input_rearranged = []
        for i in range(len(offsets) - 1):
            input_rearranged.append(input[offsets[i] : offsets[i + 1]])
        input_rearranged.append(input[offsets[-1] :])
        output = torch.zeros(len(input_rearranged), self.embedding_dim)
        for i, x in enumerate(input_rearranged):
            output[i] = self._forward(x)
        return output

import torch
import itertools
from torch.nn import Module, ParameterList, Linear, PairwiseDistance
from torch.nn.functional import leaky_relu
from torch.optim import SGD
from torch.utils.data import Dataset
from math import comb

seed = 42
torch.manual_seed(seed)

class ContrastiveLoss(Module):
    # https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        """
        Params
        ======
        margin | float
            A margin that dictates how far away negative samples should be

        p | float
        """
        super(ContrastiveLoss, self).__init__()
        self.dist_f = PairwiseDistance(p=1)
        self.margin = margin

    def forward(self, left, right, similarity):
        """
        Params
        ======
        left | tensor (n_features)
            A 'left' sample

        right | tensor (n_features)
            A 'right' sample

        similarity | [0, 1]
            The similarity between 'left' and 'right' where
                0 - Fully similar
                1 - Fully disimilar
        """
        dist = self.dist_f(left, right)
        return torch.mean(
            (1-similarity) * torch.pow(dist, 2)
          + (similarity) * torch.pow(torch.clamp(self.margin - dist, min=0.0),2)
        )

# TODO 3c388fe: Get i'th pair in lexographical order
#   Currently __getitem__ relies on the entire
#   set of possible pairs being generated which
#   is very memory inefficient.
#
#   Given i, how to calculate a, b such that (s_a, s_b)
#   is the i'th possible pair when all pairs are sorted lexographically
#
#   Example of lexographical sort:
#   [0,1,2,3] = [(0,1), (0,2), (0,3), (1, 2), (1,3), (2,3)]
#
class SiameseTrainingPairs(Dataset):
    """
    Takes `samples` and `labels` to convert it to a dataset
    for training a Siamese Network.
    """

    def __init__(self, samples, sample_info, similarity_func):
        """
        Params
        ======
        samples | tensor (n_samples, n_features)
            The samples that the SNN will train on.

        sample_info | tensor (n_samples, n_label_length)
            The labels that indicate which samples
            should be considered similar.

        similarity_func | callable( s_a, m_a, s_b, m_b )
            A function that given two tuples (s_a, m_a), (s_b, m_b),
            returns a similarity score between them.

            Here s_a, s_b is used to indicate two samples and
            m_a, m_b represent their repsective info.

            Not strictly necessary but similarity_func is treated as a function
            commutative in its arguments.
        """
        self.samples = samples
        self.sample_info = sample_info
        self.similarity_func = similarity_func

        # TODO 3c388fe
        indicies = range(0, len(samples))
        self.pairs = list(itertools.combinations(indicies, 2))

    def __len__(self):
        """
        Returns
        =======
        int
            The amount of unique pairs possible `n_samples nCr 2`
        """
        # TODO 3c388fe
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Params
        ======
        idx | int
            The idx'th pair to retreieve, these are in lexographical order
            of the samples as given by itertools.combinations.

        Returns
        =======
        (s_i, s_j, similarity_ab) | (sample, sample, float)
            where s_i, s_j are samples and similarity_ij is the similarity
            between them as specified by similarity_func.
        """
        i, j = self.pairs[idx]
        s = self.samples
        m = self.sample_info

        similarity = self.similarity_func(s[i], m[i], s[j], m[j])
        return (s[i], s[j], similarity)

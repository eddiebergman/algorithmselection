import torch
from math import comb

seed = 42
torch.manual_seed(seed)

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
        samples | torch.tensor (n_samples, n_features)
            The samples that the SNN will train on.

        sample_info | torch.tensor (n_samples, n_label_length)
            The labels that indicate which samples
            should be considered similar.

        similarity_func | callable( (s_a, m_a), (s_b, m_b) )
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
        i, j = self.pairs[i]
        s = self.samples
        m = self.info

        similarity = self.similarity_func((s[i], m[i]), (s[j], m[j]))
        return (s[i], s[j], similarity)






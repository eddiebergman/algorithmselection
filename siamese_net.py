import torch
from math import comb

seed = 42
torch.manual_seed(seed)

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

    def __len__(self):
        """ 
        Returns
        =======
        int
            The amount of unique pairs possible `n_samples nCr 2`
        """
        return comb(len(self.samples), 2)

    def __getitem__(self, idx):
        """
        Params
        ======
        idx | int
            The idx'th pair to retreieve, these are in lexographical order
            of the samples as given by itertools.combinations.

        Returns
        =======
        (s_a, s_b, similarity_ab) | (sample, sample, float)
            where s_a, s_b are samples and similarity_ab is the similarity
            between them as specified by similarity_func.
        """





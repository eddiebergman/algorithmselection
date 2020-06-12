import torch
import itertools
from torch.nn import Module, ParameterList, Linear, PairwiseDistance
from torch.nn.functional import leaky_relu
from torch.optim import Adam
from torch.utils.data import Dataset
from math import comb

seed = 42
torch.manual_seed(seed)

class ContrastiveLoss(Module):
    # https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
    """
    Contrastive loss function. Relies on using a similarity between a pair
    of samples (left, right) where
                similarity(left, right) -> 0 => Fully similar
                similarity(left, right) -> 1 => Fully disimilar

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

    def forward(self, lefts, rights, similarities):
        """
        Calculates the contrastive loss between pairs of (left, right)
        samples using the similarity as the label.

        Note:
            similarity is a measure such that
                similarity(left, right) -> 0 => Fully similar
                similarity(left, right) -> 1 => Fully disimilar
        Params
        ======
        lefts | tensor (n_batch, n_features)
            A 'left' sample

        right | tensor (n_batch, n_features)
            A 'right' sample

        similarities | tensor [0, 1]
            The similarities between pairs (left, right) where
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

class SiameseNet(Module):
    """
    Trains a fixed architecture Siamese Neural Network to cluster instances
    in feature space that are considered similar.
    """

    def __init__(self, layers):
        """
        init
        """
        # TODO doc test
        super(Net, self).__init__()
        self.layers = ParameterList(layers)

    def forward(self, x):
        """
        Gets the embedding of sample x

        Params
        ======
        x | tensor (n_features)
            The sample to embed

        Returns
        =======
        tensor (size_last_layer)
            The embedding of sample `x`
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def train(self, samples=None, samples_info=None, similarity_f=None, epochs=3,
              batch_size=128, dataset=None):
        """
        Trains the Siamese Neural Network (Snn)

        Params
        ======
        samples | tensor (n_samples, n_features)
            The samples the Snn will use to train its embedding from feature
            to embeddings space.

        samples_info | iterable (n_samples)
            Any additional meta information for each sample that factors into
            a similarity function.

        similarity_f | callable : s_a, m_a, s_b, m_b -> float
            A similarity function that takes both two samples, s_a and s_b
            along with any meta information m_a and m_b that will return a
            similarity to scale training. The model is trained to approximate
            this function by the distance in it's embeddings of s_a, s_b.

        epochs | int
            How many times to go through the datset.

        batch_size | int
            The amount of pairs to pass through the network before applying
            the learning gradient.

        dataset | torch.Dataset
            Provide a dataset to load from.
        """
        params = [samples, samples_info, similarity_f]
        assert all([p is not None for p in params]) or dataloader is not None, \
            f'Please specifiy each param `sample`, `samples_info`, `similarity_f`'

        if dataset is None:
            dataset = SiameseTrainingPairs(samples, sample_info, similarity_f)

        dataloader = DataLoader(dataset, batch_size=batch_size)

        criterion = ContrastiveLoss()
        optimizer = Adam(net.parameteres(), lr=0.005)

        for epoch in range(0, epochs):
            print(f'{epoch=}')
            epoch_loss = 0.0

            for batch, (lefts, rights, similarities) in enumerate(dataloader):
                print(f'\t{batch=}')
                optimizer.zero_grad()
                loss = criterion(lefts, rights, similarities)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f'{epoch_loss=}')

        print('finished')




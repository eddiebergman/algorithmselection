import torch
import pytest
from itertools import combinations

"""
TEST 138757229: SiameseTrainingPairs
   Checking that SiameseTrainingPairs will construct
   a dataset in lexographical order and unsure its indexing operator []
   works.

   Given a list ['a', 'b', 'c', 'd'] the sorted order would be
   [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]

   Also checked is that the similarity function works across them.
"""
from .siamese_net import SiameseTrainingPairs

class TestSiameseTrainingPairs:

    @staticmethod
    def _same_class(a, b, class_a, class_b):
        return int(class_a == class_b)

    def test_init(self):
        with pytest.raises(ValueError):
            dataset = SiameseTrainingPairs(samples=['a', 'b'],
                                           sample_info=[0, 1, 1],
                                           similarity_func=self._same_class)

    def test_len(self):
        dataset = SiameseTrainingPairs(samples=['a', 'b', 'c'],
                                       sample_info=[0, 1, 1],
                                       similarity_func=self._same_class)

        possible_pairs = [('a', 'b'), ('a', 'c'), ('b', 'c')]
        assert len(dataset) == len(possible_pairs)

    def test_getitem(self):
        samples = ['a', 'b', 'c']
        classes = [0, 1, 1]
        sim_f = self._same_class
        dataset = SiameseTrainingPairs(samples, classes, sim_f)

        # Ordered lexographically

        sample_pairs = list(combinations(samples, 2))
        class_pairs = list(combinations(classes, 2))

        similarities = [
            sim_f(a, b, class_a, class_b)
            for (a, b), (class_a, class_b) in zip(sample_pairs, class_pairs)
        ]

        for item, (a, b), sim in zip(dataset, sample_pairs, similarities):
            assert item == (a, b, sim)

"""
Test 1890846940: ContrastiveLoss

Testing contrastive loss for a few different examples. This considers all
distances to be scaled [0, 1] which is possible with a maximum bound B
such that dist(x, y) <= B.

This is more to serve as a look at the function results rather than
a serious test. We choose the points as being on a simplex which
have a bound such that dist(x, y) <= sqrt(2).

    Margin = 0.5
    Similar - similarity(x, y) = 0
        - dist(x, x) = 0
        - dist(x, y) = 0.2 < margin
        - dist(x, z) = 0.7 > margin

    Disimilar - similarity(x, y) = 1
        - dist(x, x) = 0
        - dist(x, y) = 0.2 < margin
        - dist(x, z) = 0.7 > margin

    Slightly Similar - similarity(x, y) = 0.25
        - dist(x, x) = 0
        - dist(x, y) = 0.2 < margin
        - dist(x, z) = 0.7 > margin
"""
from .siamese_net import ContrastiveLoss
import math

x = torch.Tensor([1.0, 0.0])
y = torch.Tensor([(10 - 1*math.sqrt(2)) / 10,  1*math.sqrt(2) / 10])
z = torch.Tensor([(20 - 7*math.sqrt(2)) / 20,  7*math.sqrt(2) / 20])

lefts = torch.stack([x, x, x])
rights = torch.stack([x, y, z])
print(f'{lefts=}')
print(f'{rights=}')

loss_f = ContrastiveLoss(margin=0.5)
for sim in [0.0, 1.0, 0.25]:
    length = len(lefts)
    sims = torch.tensor(sim).repeat(length)
    mean_losses = loss_f(lefts, rights, sims)
    print(f'{mean_losses=}')

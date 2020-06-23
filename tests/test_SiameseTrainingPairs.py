import pytest
from itertools import combinations
from torchsnn.snn import SiameseTrainingPairs

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



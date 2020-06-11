from siamese_net import SiameseTrainingPairs
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
samples = ['a', 'b', 'c', 'd']
classes = [1, 0, 2, 2]
same_class = lambda class_a, class_b: int(class_a == class_b)

sample_combinations = list(combinations(samples, 2))
classes_combinations = list(combinations(classes, 2))
expected_similarities = [same_class(a, b) for a, b in classes_combinations]

similarity_func = lambda s_a, m_a, s_b, m_b: same_class(m_a, m_b)
dataset = SiameseTrainingPairs(samples, classes, similarity_func)

zipped_items = list(zip(dataset, sample_combinations, expected_similarities))

# Assert length of dataset is correct
assert len(dataset) == len(sample_combinations), \
    f'{len(dataset)=} != {len(sample_combinations)}='

# Assert order of items and similarity is calculated correctly
for ds_item, correct_pair, expected_similarity in zipped_items:
    assert ds_item == (expected := (*correct_pair, expected_similarity)), \
        f'{ds_item=} != {expected=}'

# Assert indexing works as required
for i in range(0, len(sample_combinations)):
    ds_item = dataset[i]
    expected = (*sample_combinations[i], expected_similarities[i])
    assert ds_item == expected, f'{ds_item=} != {expected=}'

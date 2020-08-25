import pytest
import torch
from torchsnn.snn import SiameseNet

class TestSiameseNet:

    def _layers(self):
        """ Default architecture """
        return [
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1)
        ]

    def _data(self):
        """ Returns xs, ys for testing """
        return (
            torch.tensor([[1.0, 1.0], [10.0, 1.0], [1.0, 10.0]]),
            torch.tensor([1, 1, 0])
        )

    def test_init_success(self):
        """ Test parameters are set """
        architecture = self._layers()
        snn = SiameseNet(architecture)

        assert snn.parameters() is not None

    def test_init_empty_layers(self):
        """ Test for empty layers error """
        architecture = []
        with pytest.raises(ValueError):
            snn = SiameseNet(architecture)

    def test_forward(self):
        """ Test forwarding transforms points to new locations """
        architecture = self._layers()
        snn = SiameseNet(architecture)

        xs = torch.Tensor([1.0, 1.0])
        embedded_xs = snn.forward(xs)
        assert not torch.equal(xs, embedded_xs)

    def test_train(self):
        """
        Test that the training has same class points move closer
        and different class points move further
        """
        architecture = self._layers()
        snn = SiameseNet(architecture)

        xs, ys = self._data()
        same_class = lambda a, b, class_a, class_b: \
             0 if class_a == class_b else 1

        untrained_xs = snn.forward(xs)
        snn.train(xs, ys, similarity_f=same_class, verbose=False)
        trained_xs = snn.forward(xs)

        # Check instance 0 and 1 are brought closer together as same class
        dist_untrained = torch.dist(untrained_xs[0] , untrained_xs[1]).item()
        dist_trained = torch.dist(trained_xs[0], trained_xs[1]).item()
        assert dist_untrained >= dist_trained

        # Check instance 0 and 2 are brough further apart as not same class
        dist_untrained = torch.dist(untrained_xs[0] , untrained_xs[2]).item()
        dist_trained = torch.dist(trained_xs[0], trained_xs[2]).item()
        assert dist_untrained <= dist_trained

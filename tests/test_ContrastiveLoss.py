import pytest
import torch
import math
from torchsnn.snn import ContrastiveLoss

class TestContrastiveLoss:

    def test_init(self):
        with pytest.raises(ValueError):
            ContrastiveLoss(margin=-1)

    def test_forward(self):
        # TODO: The outputs of the function shold be tested properly
        loss_f = ContrastiveLoss(margin=0.5)

        x = torch.Tensor([1.0, 0.0])
        y = torch.Tensor([(10 - 1*math.sqrt(2)) / 10,  1*math.sqrt(2) / 10])
        z = torch.Tensor([(20 - 7*math.sqrt(2)) / 20,  7*math.sqrt(2) / 20])

        lefts = torch.stack([x, x, x])
        rights = torch.stack([x, y, z])

        for sim in [0.0, 1.0, 0.25]:
            length = len(lefts)
            sims = torch.tensor(sim).repeat(length)
            mean_losses = loss_f(lefts, rights, sims)
            assert mean_losses is not None


import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def loss(self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
        """Given a tensor of inputs and targets, compute the average cross-entropy
        loss across examples.

        Args:
            inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
                unnormalized logit of jth class for the ith example.
            targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
                Each value must be between 0 and `num_classes - 1`.

        Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
        """
        target_p = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
        logsumexp = torch.logsumexp(inputs, dim=-1)
        loss = torch.mean(-target_p + logsumexp)
        return loss

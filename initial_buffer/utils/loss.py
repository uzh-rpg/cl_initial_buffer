import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class SoftNearestNeighborLoss(_Loss):
    def __init__(self, temperature=0.5) -> None:
        """
        :param temperature: temperature for penalizing the negative embedding distance
        """
        super().__init__()
        self.temperature = torch.tensor(temperature)

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the cosine similarity between x and y, the range is [-1, 1]

        Args:
            x: input tensor
            y: input tensor
        """
        return F.cosine_similarity(F.normalize(x, p=2), F.normalize(y, p=2))

    def forward(self, input_batch: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_batch: input batch
            pos: positive sample
            neg: negative sample
            pos_similarity: positive similarity
            neg_similarity: negative similarity
        """
        assert input_batch.dim() == 2
        assert pos.dim() == 2
        assert neg.dim() == 2

        if self.temperature.device != input_batch.device:
            self.temperature = self.temperature.to(input_batch.device)

        num_similarity = F.cosine_similarity(F.normalize(input_batch, p=2), F.normalize(pos, p=2))
        den_similarity = F.cosine_similarity(F.normalize(input_batch, p=2), F.normalize(neg, p=2))

        num = torch.mean(torch.exp(num_similarity / self.temperature), dim=0)
        den = torch.mean(torch.exp(den_similarity / self.temperature), dim=0) + num

        loss = -torch.log(num / den)

        return loss.mean()

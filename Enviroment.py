import torch
from torch import Tensor


class Environment:

    def __init__(self):

        self.state = torch.tensor([])

        self.history = []

        self.deltaT = 0.1

        ...

    def next(self, action: Tensor) -> Tensor:
        ...

    def run(self):
        ...
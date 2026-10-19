
from Games.HitBall import HitBall

from Agent import SearchNode

import torch

s = SearchNode(1, torch.zeros(1), 1, 1)

print(s.getRectVariation())


game = HitBall()

game.run()




import torch
from torch import Tensor

from Enviroment import Environment

from Agent import Agent


def normalizeAction(action: Tensor) -> Tensor:

    pushD = action[0]

    if pushD > 0:
        return torch.tensor([1.0], device=action.device)
    if pushD < 0:
        return torch.tensor([-1.0], device=action.device)
    return torch.tensor([0.0], device=action.device)

class Timmy(Agent):
    def __init__(self):

        super().__init__(1, 1)

class HitBall(Environment):
    def __init__(self):
        super().__init__()

        self.state = torch.tensor([5])

        self.actor = Timmy()


    def next(self, action: Tensor) -> Tensor:

        pushD = normalizeAction(action)[0]

        return torch.add(pushD, self.state)

    def run(self):

        for _ in range(10):

            agentAction = self.actor.act(self.state)

            print("action")
            print(agentAction)

            self.state = torch.add(self.state, normalizeAction(agentAction))

            self.actor.logAct(self.state)

        print(self.actor.jacobian.forward(torch.cat((self.state, agentAction))))

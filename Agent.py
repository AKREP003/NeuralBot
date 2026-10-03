import torch
from torch import Tensor

class Action:
    def __init__(self,
                 prevState: Tensor = torch.zeros(0),
                 action: Tensor = torch.zeros(0),
                 nextState: Tensor = torch.zeros(0)):

        self.prevState = prevState

        self.action = action

        self.nextState = nextState

    def initState(self):
        return torch.cat((self.prevState, self.action), dim=0)


class Agent:
    def __init__(self, stateD: int, actionD: int ) -> None:

        self.stateD = stateD

        self.actionD = actionD

        self.memory: list[Action] = []

        self.memory.append(Action())

        self.idealState: Tensor = torch.tensor(stateD)

        self.jacobian = torch.nn.Sequential(
            torch.nn.Linear(stateD + actionD,
                            stateD)
        )

        self.learning_rate = 1e-3

        self.jacobianOptimizer = torch.optim.RMSprop(self.jacobian.parameters(),
                                                     lr=self.learning_rate)

        self.jacobianTrainIter: int = 1000

        self.jacobianLoss = torch.nn.MSELoss(reduction='sum')

    def judge(self, state: Tensor) -> Tensor:

        return torch.sub(state, self.idealState).abs().sum()

    def updateJacobian(self) -> None:

        inp = self.memory[-1].initState()

        print("inp")

        print(inp)

        out = self.memory[-1].nextState

        print("out")

        print(out)


        for t in range(self.jacobianTrainIter):

            predState = self.jacobian.forward(inp)

            loss = self.jacobianLoss(predState, out)

            if t % 100 == 99:
                print(t, loss.item())

            self.jacobianOptimizer.zero_grad()

            loss.backward()

            self.jacobianOptimizer.step()

        print(self.jacobian[0].weight)



    def logAct(self, obsState: Tensor) -> None:

        if not self.memory: return

        self.memory[-1].nextState = obsState.clone()

        self.updateJacobian()

        self.memory.append(Action())

    def act(self, state: Tensor) -> Tensor:

        self.memory[-1].prevState = state.clone()
        self.memory[-1].action = torch.ones(self.actionD)

        return self.memory[-1].action



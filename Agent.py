import torch
from torch import Tensor

class Action:
    def __init__(self,
                 prevState: Tensor = torch.zeros(0),
                 action: Tensor = torch.zeros(0),
                 stateChange: Tensor = torch.zeros(0)):

        self.prevState = prevState

        self.action = action

        self.stateChange = stateChange

    def initState(self):
        return torch.cat((self.prevState, self.action), dim=0)

    def clone(self):
        return Action(self.prevState.clone(), self.action.clone(), self.stateChange.clone())

class Agent:
    def __init__(self, stateD: int, actionD: int ) -> None:

        self.stateD = stateD

        self.actionD = actionD

        self.memory: list[Action] = []

        self.memory.append(Action())

        self.idealState: Tensor = torch.zeros(stateD)

        self.jacobian = torch.nn.Sequential(
            torch.nn.Linear(stateD + actionD,
                            stateD),
            torch.nn.Linear(stateD,
                            stateD),
        )

        self.learning_rate = 1e-3

        self.jacobianOptimizer = torch.optim.RMSprop(self.jacobian.parameters(),
                                                     lr=self.learning_rate)

        self.jacobianTrainIter: int = 1000

        self.jacobianLoss = torch.nn.MSELoss(reduction='sum')

        self.decideIter: int = 1000

        self.pastRecollection: int = 2


    def judge(self, state: Tensor) -> Tensor:

        return self.jacobianLoss(state, self.idealState)

    def getMemRange(self):

        memSize = len(self.memory)


        return (max(0, memSize - self.pastRecollection),
                memSize - 1
                )

    def recollection(self) -> None:

        remaining = self.jacobianTrainIter

        memRange = self.getMemRange()

        index = memRange[0]

        while remaining > 0:

            if index > memRange[1]: index = memRange[0]

            inp = self.memory[index].initState()

            out = self.memory[index].stateChange

            self.updateJacobian(inp, out)

            index += 1

            remaining -= 1



    def updateJacobian(self, inp:Tensor, out:Tensor) -> None:

        #inp = self.memory[-1].initState()

        #out = self.memory[-1].nextState

        predState = self.jacobian.forward(inp)

        loss = self.jacobianLoss(predState, out)

        self.jacobianOptimizer.zero_grad()

        loss.backward()

        self.jacobianOptimizer.step()

    def logAct(self, obsState: Tensor) -> None:

        print("obsState")
        print(obsState)

        if not self.memory: return

        self.memory[-1].stateChange = torch.sub(obsState, self.memory[-1].prevState)

        self.recollection()

        self.memory.append(Action())

    def decide(self, state: Tensor) -> Tensor:

        actionBuffer = torch.zeros(self.actionD, requires_grad=True)

        opt = torch.optim.RMSprop([actionBuffer], lr=self.learning_rate)

        for _ in range(self.decideIter):

            potential : Action = Action()

            potential.prevState = state.clone()

            potential.action = actionBuffer

            predChange =  self.jacobian.forward(potential.initState())

            predState = torch.add(state, predChange)

            loss = self.judge(predState)

            opt.zero_grad()

            loss.backward()

            opt.step()

        return actionBuffer

    def act(self, state: Tensor) -> Tensor:

        if not self.memory: self.memory.append(Action())

        self.memory[-1].prevState = state.clone()

        #self.memory[-1].action = torch.ones(self.actionD)

        self.memory[-1].action = self.decide(state)

        return self.memory[-1].action



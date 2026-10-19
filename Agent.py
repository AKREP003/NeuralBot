import types

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
        return Action(torch.clone(self.prevState), torch.clone(self.action), torch.clone(self.stateChange))

    def __str__(self):

        return f"{self.prevState.__str__()} + {self.action.__str__()} -> {self.stateChange.__str__()}"


class SearchNode:
    def __init__(
        self,
        actionD: int,
        action: Tensor,
        delta: float,
        searchCount: int = 1,
    ):

        self.actionD = actionD

        self.action = action
        self.delta = delta
        self.searchCount = searchCount

    def clone(self):
        return SearchNode(
            actionD=self.actionD,
            action=torch.clone(self.action),
            delta=self.delta,
            searchCount=self.searchCount,
        )

    def getRectVariation(self) -> list[Tensor]:
        return self.singleDeviation(0)

    def singleDeviation(self, index: int) -> list[Tensor]:

        if index >= self.actionD: return [self.action]

        variations = []

        for dist in range(-self.searchCount, self.searchCount + 1):
            node = self.clone()

            change = dist * self.delta

            with torch.no_grad():
                node.action[index] += change

            variations.extend(node.singleDeviation(index + 1))

        return variations




class SearchSpace:
    def __init__(self,
                 stateD: int,
                 actionD: int,
                 state: Tensor,
                 delta: int,
                 pred,
                 loss,
                 ):

        self.stateD = stateD
        self.actionD = actionD
        self.state = state
        self.delta = delta
        self.loss = loss
        self.pred = pred

    def actionList(self, state: Tensor, action: Tensor) -> list[Action]:

        node: SearchNode = SearchNode(self.actionD, action, self.delta, 1)

        actVecList: list[Tensor] = node.getRectVariation()

        actListBuffer: list[Action] = []

        for act in actVecList:

            actionObj: Action = Action()

            actionObj.action = act

            actionObj.prevState = torch.clone(state)

            initState = actionObj.initState()

            actionObj.stateChange = self.pred(initState)

            actListBuffer.append(actionObj)

        return actListBuffer


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

    def changePred(self, initState: Tensor) -> Tensor:

        return self.jacobian.forward(initState)


    def decide(self, state: Tensor) -> Tensor:

        space: SearchSpace = SearchSpace(self.stateD, self.actionD, state, 1, self.changePred, self.judge)

        for k in space.actionList(state, torch.zeros(self.stateD)):

            print(k)



        actionBuffer = torch.zeros(self.actionD, requires_grad=True)

        opt = torch.optim.RMSprop([actionBuffer], lr=self.learning_rate)

        for _ in range(self.decideIter):

            potential : Action = Action()

            potential.prevState = torch.clone(state)

            potential.action = actionBuffer

            predChange = self.changePred(potential.initState())

            predState = torch.add(state, predChange)

            loss = self.judge(predState)

            opt.zero_grad()

            loss.backward()

            opt.step()

        return actionBuffer

    def act(self, state: Tensor) -> Tensor:

        if not self.memory: self.memory.append(Action())

        self.memory[-1].prevState = torch.clone(state)

        #self.memory[-1].action = torch.ones(self.actionD)

        self.memory[-1].action = self.decide(state)

        return self.memory[-1].action



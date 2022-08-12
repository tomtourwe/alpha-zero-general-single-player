import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

class PlanningRepresentation:

    def __init__(self, machines, timesteps, domainactions, rewardstrategy, legal_actions, current_domainaction):
        self.machines = machines
        self.timesteps = timesteps
        self.schedule = np.zeros((self.machines, self.timesteps), dtype=int)
        self.domainactions = domainactions
        self.rewardstrategy = rewardstrategy
        self.legal_actions = legal_actions
        self.current_domainaction = current_domainaction
        # self.legal_actions = [i for i in range(self.machines * self.timesteps)]
        # self.current_domainaction = 0

    def __getitem__(self, index): 
        return self.schedule[index]

    def _action_to_move(self, action):
        return (action % self.machines, int(action / self.machines))

    def _move_to_action(self, move):
        machine, time = move
        return machine + (self.machines * time)

    def _valid_action(self, action, domainaction):
        machine, time = self._action_to_move(action)
        duration = domainaction.duration
        if time + duration < self.timesteps:
            for t in range(time, time+duration):
                if self.schedule[machine, t] != 0:
                    return False
            return True
        return False

    def is_done(self):
        return self.current_domainaction == len(self.domainactions)

    def get_legal_moves(self):
        if not self.is_done():
            next_domainaction = self.domainactions[self.current_domainaction]
            return set([a for a in self.legal_actions if self._valid_action(a, next_domainaction)])
        else:
            return set()

    def execute_move(self, action):
        (machine,timestep) = self._action_to_move(action)
        domainaction = self.domainactions[self.current_domainaction]
        duration = domainaction.duration
        for t in range(timestep, timestep+duration):
            self.schedule[machine,t] = domainaction.urn
            self.legal_actions.remove(self._move_to_action((machine, t)))
        self.current_domainaction += 1

    def compute_reward(self):
        return self.rewardstrategy.compute_reward(self.schedule)



@dataclass(frozen=True)
class DomainAction:
    duration: int
    urn: int
    deadline: int = 0 # represents that the action should be scheduled before `deadline`
    start: int = -1 # represents a fixed action to be scheduled at `start`
    machine: int = -1 # represents a fixed action to be scheduled on `machine`

    def __repr__(self):
        return f"Action(urn={self.urn},duration={self.duration},deadline={self.deadline},start={self.start},machine={self.machine})"

class RewardStrategy:

    def __init__(self, min_reward):
        self.min_reward = min_reward

    def get_min_reward(self):
        return self.min_reward

    @abstractmethod
    def compute_reward(self, schedule):
        pass

class RelativeProductRewardStrategy(RewardStrategy):
    def __init__(self, min_reward):
        super(RelativeProductRewardStrategy, self).__init__(min_reward)

    # consider the number of free time slots before the last scheduled action per machine
    # compute the product over all the machines. 
    def compute_reward(self, schedule):
        return -np.product([1/(np.sum(row[:np.max(np.nonzero(row))+1] != 0)/(np.max(np.nonzero(row))+1)) if np.any(row != 0) else 1 for row in schedule])

class MinSpanTimeRewardStrategy(RewardStrategy):

    def __init__(self, min_reward):
        super(MinSpanTimeRewardStrategy, self).__init__(min_reward)

    def compute_reward(self, schedule):
        _, col_idxs = np.nonzero(schedule)
        return -max(col_idxs) if col_idxs.size != 0 else self.get_min_reward()

class JustInTimeRewardStrategy(RewardStrategy):

    def __init__(self, min_reward):
        super(JustInTimeRewardStrategy, self).__init__(min_reward)

    def compute_reward(self, schedule):
        row_idxs, col_idxs = np.nonzero(schedule)
        actions = [(schedule[row,col], col) for row, col in zip(row_idxs, col_idxs)]
        res = self.get_min_reward()
        if actions: # if actions is empty, np.mean raises a warning, so we check for it.
            d = {}
            for action, t in actions:
                d[action] = max(d.get(action, -float('inf')), t)
            # res = np.mean([-np.abs(t - action.deadline) for action, t in d.items()])
            res = np.min([(t - action.deadline) if t < action.deadline else 3 * (action.deadline - t) 
                            for action, t in d.items()])
        return res
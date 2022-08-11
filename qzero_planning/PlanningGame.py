from Game import Game
from .PlanningLogic import PlanningRepresentation
import numpy as np
from sympy.utilities.iterables import multiset_permutations

class PlanningGame(Game):
    def __init__(self, machines, timesteps, domainactions, rewardstrategy):
        super(PlanningGame, self).__init__()
        self.machines = machines
        self.timesteps = timesteps
        self.domainactions = domainactions
        self.rewardstrategy = rewardstrategy
        self.legal_actions = [i for i in range(machines * timesteps)]
        self.current_domainaction = 0

    def _make_representation(self):
        return PlanningRepresentation(self.machines,
                                        self.timesteps, 
                                        self.domainactions, 
                                        self.rewardstrategy,
                                        list(np.copy(self.legal_actions)),
                                        self.current_domainaction)

    def get_copy(self):
        c = PlanningGame(self.machines, 
                            self.timesteps,
                            self.domainactions,
                            self.rewardstrategy)
        c.legal_actions = list(np.copy(self.legal_actions))
        c.current_domainaction = self.current_domainaction
        return c

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        r = self._make_representation()
        return r.schedule

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.machines, self.timesteps)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return (self.machines * self.timesteps) + 1

    def getNextState(self, board, action):
        """
        Input:
            board: current board
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        r = self._make_representation()
        r.schedule = np.copy(board)
        r.execute_move(action)
        self.current_domainaction = r.current_domainaction
        self.legal_actions = r.legal_actions
        return r.schedule

    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        r = self._make_representation()
        r.schedule = np.copy(board)
        legalMoves =  r.get_legal_moves()
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended, reward otherwise. 
               
        """
        r = self._make_representation()
        r.schedule = np.copy(board)
        return r.compute_reward() if r.is_done() else None

    def getCanonicalForm(self, board):
        """
        Input:
            board: current board

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    # def _unique_permutations(x):
    #     # ys = list of (row_idx, boolean) indicating whether 
    #     # or not the row at index idx has non-zero elements
    #     ys = list(zip(range(len(x)), np.any(x != 0, axis=1))) 
    #     # sort the list so that the last elemant can be (row_idx, False)
    #     # IF there is a row with only zeros
    #     ys = sorted(ys, key=lambda x: x[1], reverse=True)
    #     # keep the idx if (idx, True) else keep the idx of the last element
    #     idxs = [x[0] if x[1] else ys[-1][0] for x in ys]
    #     # compute the permutations without duplicates and map back to the input
    #     return [x[p] for p in multiset_permutations(idxs)]

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # return board.tobytes()s
        return "\n".join([f"{x}:[{','.join([str(y) for y in board[x]])}]" for x in range(len(board))])
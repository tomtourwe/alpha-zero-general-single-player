import logging

import coloredlogs

from Coach import Coach

from qzero_planning.NNet import NNetWrapper as pnn
from qzero_planning.PlanningGame import PlanningGame
from qzero_planning.PlanningLogic import DomainAction, MinSpanTimeRewardStrategy, RelativeProductRewardStrategy

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    domainactions = [DomainAction(urn=1, duration=2), DomainAction(urn=2, duration=2),
                     DomainAction(urn=3, duration=1), DomainAction(urn=4, duration=1),
                     DomainAction(urn=5, duration=2), DomainAction(urn=6, duration=1)]
    machines = 6
    timesteps = 6

    log.info(f'Loading {PlanningGame.__name__}...')
    # g = PlanningGame(machines=machines, timesteps=timesteps, domainactions=domainactions,rewardstrategy=MinSpanTimeRewardStrategy(-((machines*timesteps) + 1)))
    g = PlanningGame(machines=machines, timesteps=timesteps, domainactions=domainactions,rewardstrategy=RelativeProductRewardStrategy(-((machines**timesteps)+1)))
    
    log.info('Loading %s...', pnn.__name__)
    nnet = pnn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()

from Parameters import EnviromentParameters
from pyAgrum.lib.bn2roc import showROC
import pyAgrum.lib.notebook as gnb
import pyAgrum as gum
from enum import Enum
from Utils.PrintUtils import print_big


class BN_Algorithm(Enum):
    HillClimbing = 1
    LocalSearch = 2
    ThreeOffTwo = 3
    MIIC = 4


def learnBN(file_path: str, algorithm: BN_Algorithm = BN_Algorithm.HillClimbing):
    learner = gum.BNLearner(file_path)

    if(algorithm == BN_Algorithm.HillClimbing):
        print_big("Selecting Greedy Hill Climbing Algorithm")
        learner.useGreedyHillClimbing()

    elif(algorithm == BN_Algorithm.LocalSearch):
        print_big("Selecting Local Search Algorithm")
        bn = learner.useLocalSearchWithTabuList()

    elif(algorithm == BN_Algorithm.ThreeOffTwo):
        print_big("Selecting 3Off2 Algorithm")
        learner.use3off2()

    elif(algorithm == BN_Algorithm.MIIC):
        print_big("Selecting MIIC Algorithm")
        learner.useMIIC()

    else:
        raise Exception('Not supported algorithm')

    bn = learner.learnBN()
    essencGraph = gum.EssentialGraph(bn)
    infoBN = gnb.getInformation(bn, size=EnviromentParameters.default_graph_size)

    return [bn, infoBN, essencGraph]

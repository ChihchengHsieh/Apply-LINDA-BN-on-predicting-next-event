from enum import Enum


class SelectableDatasets(Enum):

    #############
    # PM
    #############

    BPI2012 = "BPI2012"
    Helpdesk = "Helpdesk"

    #############
    # Medical
    #############
    Diabetes = "Diabetes"
    BreastCancer = "BreastCancer"


class SelectableLoss(Enum):
    CrossEntropy = "CrossEntropy"
    BCE = "BCE"


class SelectableModels(Enum):
    BaseLineLSTMModel = "BaseLineLSTMModel"
    BaseNNModel = "BaseNNModel"


class SelectableOptimizer(Enum):
    Adam = "Adam"
    SGD = "SGD"


class SelectableLrScheduler(Enum):
    StepScheduler = "StepScheduler"
    NotUsing = "NotUsing"


class PreprocessedDfType(Enum):
    Pickle = "Pickle"


class ActivityType(Enum):
    O = "O"
    A = "A"
    W = "W"


class PermuatationSampleDist(Enum):
    Normal = "Normal"
    Uniform = "Uniform"

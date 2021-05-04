from enum import Enum

class SelectableDatasets(Enum):
    BPI2012 = "BPI2012"
    Helpdesk = "Helpdesk"

class SelectableLoss(Enum):
    CrossEntropy = "CrossEntropy"

class SelectableModels(Enum):
    BaseLineLSTMModel = "BaseLineLSTMModel"

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


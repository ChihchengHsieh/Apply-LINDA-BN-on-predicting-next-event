from enum import Enum

class SelectableDatasets(Enum):
    BPI2012 = 1
    Helpdesk = 2

class SelectableLoss(Enum):
    CrossEntropy = 1

class SelectableModels(Enum):
    BaseLineLSTMModel = 1

class SelectableOptimizer(Enum):
    Adam = 1
    SGD = 2

class SelectableLrScheduler(Enum):
    StepScheduler = 1
    NotUsing = 2

class PreprocessedDfType(Enum):
    Pickle = 1
    

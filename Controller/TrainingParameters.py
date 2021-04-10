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

class TrainingParameters:
    '''
    LSTM baseline model parameters
    '''
    bpi_2012_path = '../Data/event_logs/BPI_Challenge_2012.xes'

    dataset: SelectableDatasets = SelectableDatasets.BPI2012
    model: SelectableModels = SelectableModels.BaseLineLSTMModel
    optimizer: SelectableOptimizer = SelectableOptimizer.Adam
    loss: SelectableLoss = SelectableLoss.CrossEntropy

    stop_epoch: int = 1

    class OptimizerParameters:
        learning_rate: float = 0.005
        l2 = 0.1

    class BaselineLSTMModelParameters:
        embedding_dim: int = 64
        lstm_hidden: int = 32
        dropout: float = .8
        num_lstm_layers: int = 2




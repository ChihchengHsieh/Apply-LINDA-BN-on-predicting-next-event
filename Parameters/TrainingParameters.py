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


class SelectableLrScheduler(Enum):
    StepScheduler = 1
    NotUsing = 2


class TrainingParameters:
    '''
    Storing the parameters for controlling the training.
    '''

    # Move it to dataset path class
    bpi_2012_path: str = '../Data/event_logs/BPI_Challenge_2012.xes'

    dataset: SelectableDatasets = SelectableDatasets.BPI2012
    model: SelectableModels = SelectableModels.BaseLineLSTMModel
    optimizer: SelectableOptimizer = SelectableOptimizer.Adam
    loss: SelectableLoss = SelectableLoss.CrossEntropy

    stop_epoch: int = 1
    batch_size: int = 32
    # Remaining will be used for validation.
    train_test_split_portion = [0.8, 0.1]

    verbose_freq: int = 20
    run_validation_freq: int = 40

    class OptimizerParameters:
        learning_rate: float = 0.005
        l2: float = 0.1
        lr_scheduler_step: int = 200
        lr_scheduler_gamma: float = .85
        scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler

    class BaselineLSTMModelParameters:
        embedding_dim: int = 16 # 128
        lstm_hidden: int = 32 # 256
        dropout: float = .1
        num_lstm_layers: int = 1 # 2

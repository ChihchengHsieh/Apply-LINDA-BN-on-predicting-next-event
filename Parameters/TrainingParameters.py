from enum import Enum
from Utils.SaveUtils import get_json_dict
import json


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


class PreprocessedDfType(Enum):
    Pickle = 1
    HDF5 = 2


class TrainingParameters(object):
    '''
    Storing the parameters for controlling the training.
    '''

    bpi_2012_path: str = '../Data/event_logs/BPI_Challenge_2012.xes'
    preprocessed_bpi_2012_folder_path = '../Data/preprocessed/BPI_Challenge_2012'
    preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.HDF5

    # load_model_folder_path: str = "SavedModels/2021-04-12 03:06:47.329664" # Set to None to not loading pre-trained model.
    # Set to None to not loading pre-trained model.
    load_model_folder_path: str = None
    load_optimizer: bool = False

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

    class OptimizerParameters(object):
        learning_rate: float = 0.005
        l2: float = 0.1
        lr_scheduler_step: int = 200
        lr_scheduler_gamma: float = .85
        scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler

    class BaselineLSTMModelParameters(object):
        embedding_dim: int = 16  # 128
        lstm_hidden: int = 32  # 256
        dropout: float = .1
        num_lstm_layers: int = 1  # 2

    @staticmethod
    def save_parameters_json__(path: str):
        parameters_dict = get_json_dict(TrainingParameters)
        with open(path, 'w') as output_file:
            json.dump(parameters_dict, output_file, indent="\t")

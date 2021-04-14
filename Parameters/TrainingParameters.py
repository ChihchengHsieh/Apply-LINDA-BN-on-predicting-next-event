import json
from Utils.SaveUtils import get_json_dict
from Parameters.Enums import (
    PreprocessedDfType,
    SelectableDatasets,
    SelectableLoss,
    SelectableLrScheduler,
    SelectableModels,
    SelectableOptimizer,
)


class TrainingParameters(object):
    """
    Storing the parameters for controlling the training.
    """

    # will be save as the name.
    parameters_save_file_name__ = "parameters.json"

    ######################################
    # Parameters for training
    ######################################

    bpi_2012_path: str = "./datasets/event_logs/BPI_Challenge_2012.xes"
    preprocessed_bpi_2012_folder_path = "./datasets/preprocessed/BPI_Challenge_2012"
    preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.HDF5

    # Set to None to not loading pre-trained model.
    # load_model_folder_path: str = "SavedModels/2021-04-13 01:07:49.273685"
    # Set to None to not loading pre-trained model.
    load_model_folder_path: str = None
    load_optimizer: bool = True

    dataset: SelectableDatasets = SelectableDatasets.BPI2012
    model: SelectableModels = SelectableModels.BaseLineLSTMModel
    optimizer: SelectableOptimizer = SelectableOptimizer.Adam
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    stop_epoch: int = 100
    batch_size: int = 64
    train_test_split_portion = [0.8, 0.1]  # Remaining will be used for validation.
    verbose_freq: int = 100  # in step
    run_validation_freq: int = 200  # in step

    dataset_split_seed = 12345

    max_eos_predicted_length = 50

    class OptimizerParameters(object):
        """
        It will be override once you have load_model and load_optimizer = True
        """

        learning_rate: float = 0.005
        l2: float = 0.000001

        ## Scheduler
        scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler
        lr_scheduler_step: int = 1000
        lr_scheduler_gamma: float = 0.85
        SGD_momentum = 0.9

    class BaselineLSTMModelParameters(object):
        """
        It will be override once you have load_model
        """

        embedding_dim: int = 256  # 128
        lstm_hidden: int = 512  # 256
        dropout: float = 0.2
        num_lstm_layers: int = 2  # 2

    @staticmethod
    def save_parameters_json__(path: str):
        parameters_dict = get_json_dict(TrainingParameters)
        with open(path, "w") as output_file:
            json.dump(parameters_dict, output_file, indent="\t")

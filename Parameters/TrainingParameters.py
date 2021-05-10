import json
from Utils.SaveUtils import get_json_dict
from Parameters.Enums import (
    SelectableDatasets,
    SelectableLoss,
    SelectableLrScheduler,
    SelectableModels,
    SelectableOptimizer,
    ActivityType
)


class TrainingParameters(object):
    """
    Storing the parameters for controlling the training.
    """

    # will be save as the name.
    parameters_save_file_name__ = "parameters.json"
    
    #########################
    # Load
    ########################

    # load_model_folder_path: str = "SavedModels/0.7237_Diabetes_BaseNNModel_2021-05-10 02:57:11.018429" # Set to None to not loading pre-trained model.
    load_model_folder_path: str = None # Set to None to not loading pre-trained model.
    load_optimizer: bool = True

    ######################################
    # Selectables
    #####################################
    dataset: SelectableDatasets = SelectableDatasets.Diabetes
    model: SelectableModels = SelectableModels.BaseNNModel
    loss: SelectableLoss = SelectableLoss.BCE
    optimizer: SelectableOptimizer = SelectableOptimizer.Adam

    ######################################
    # Count 
    ######################################
    stop_epoch: int = 10
    batch_size: int = 128
    verbose_freq: int = 250 # in step
    run_validation_freq: int = 500  # in step

    ######################################
    # Dataset
    ######################################
    train_test_split_portion = [0.8, 0.1] # Remaining will be used for validation.
    dataset_split_seed = 1234

    class BPI2012(object):
        BPI2012_include_types = [ActivityType.A]

    class OptimizerParameters(object):
        """
        It will be override once you have load_model and load_optimizer = True
        """
        learning_rate: float = 0.005
        l2: float = 0.001

        # Scheduler
        scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler
        lr_scheduler_step: int = 800
        lr_scheduler_gamma: float = 0.8
        SGD_momentum = 0.9

    class BaseNNModelParams(object):
        hidden_dim = [8]*8
        dropout = .2

    class BaselineLSTMModelParameters(object):
        """
        It will be override once you have load_model
        """
        embedding_dim: int = 256  # 128
        lstm_hidden: int = 512  # 256
        dropout: float = 0.2
        num_lstm_layers: int = 2  # 2

    ########################
    # Others
    ########################
    max_eos_predicted_length = 50
    plot_cm = False

    @staticmethod
    def save_parameters_json__(path: str):
        parameters_dict = get_json_dict(TrainingParameters)
        with open(path, "w") as output_file:
            json.dump(parameters_dict, output_file, indent="\t")

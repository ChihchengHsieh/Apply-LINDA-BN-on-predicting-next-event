import json
from Utils.SaveUtils import get_json_dict
from Parameters.Enums import PreprocessedDfType, SelectableDatasets, SelectableLoss, SelectableLrScheduler, SelectableModels, SelectableOptimizer


class TrainingParameters(object):
    '''
    Storing the parameters for controlling the training.
    '''
    # will be save as the name.
    parameters_save_file_name__ = "parameters.json"

    ######################################
    # Parameters for training
    ######################################

    bpi_2012_path: str = './datasets/event_logs/BPI_Challenge_2012.xes'
    preprocessed_bpi_2012_folder_path = './datasets/preprocessed/BPI_Challenge_2012'
    preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.HDF5

    # Set to None to not loading pre-trained model.
    # load_model_folder_path: str = "SavedModels/2021-04-13 01:07:49.273685"
    # Set to None to not loading pre-trained model.
    load_model_folder_path: str = None
    load_optimizer: bool = True

    dataset: SelectableDatasets = SelectableDatasets.BPI2012
    model: SelectableModels = SelectableModels.BaseLineLSTMModel
    optimizer: SelectableOptimizer = SelectableOptimizer.SGD
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    stop_epoch: int = 1
    batch_size: int = 32
    # Remaining will be used for validation.
    train_test_split_portion = [0.8, 0.1]
    verbose_freq: int = 20
    run_validation_freq: int = 40

    class OptimizerParameters(object):
        '''
        It will be override once you have load_model and load_optimizer = True
        '''
        learning_rate: float = 0.005
        l2: float = 0.1

        ## Scheduler
        scheduler: SelectableLrScheduler = SelectableLrScheduler.StepScheduler
        lr_scheduler_step: int = 200
        lr_scheduler_gamma: float = .85
        SGD_momentum = .9

    class BaselineLSTMModelParameters(object):
        '''
        It will be override once you have load_model
        '''
        embedding_dim: int = 128  # 128
        lstm_hidden: int = 256  # 256
        dropout: float = .1
        num_lstm_layers: int = 2  # 2

    @staticmethod
    def save_parameters_json__(path: str):
        parameters_dict = get_json_dict(TrainingParameters)
        with open(path, 'w') as output_file:
            json.dump(parameters_dict, output_file, indent="\t")

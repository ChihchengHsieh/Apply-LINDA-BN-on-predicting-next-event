from Parameters.Enums import PreprocessedDfType, SelectableDatasets, SelectableLoss, SelectableModels, 


class PredictingParameters(object):
    '''
    Storing the parameters for controlling the predicting.
    '''
    # will be save as the name.
    parameters_save_file_name__ = "parameters.json"

    ######################################
    # Parameters for predicting
    ######################################

    bpi_2012_path: str = './datasets/event_logs/BPI_Challenge_2012.xes'
    preprocessed_bpi_2012_folder_path = './datasets/preprocessed/BPI_Challenge_2012'
    preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.HDF5

    load_model_folder_path: str = "./SavedModels/2021-04-14 01:25:51.801160"  # Must set

    max_eos_predicted_length = 50

    standard_dataset: SelectableDatasets = SelectableDatasets.BPI2012
    model: SelectableModels = SelectableModels.BaseLineLSTMModel
    # Loss function for measuring the loss on evaluation
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    batch_size: int = 32  # batch_size when running evaluation on the

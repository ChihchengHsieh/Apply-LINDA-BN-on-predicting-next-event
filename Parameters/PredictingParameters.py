from Parameters.Enums import PreprocessedDfType, SelectableLoss, SelectableModels


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
    preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.Pickle

    dataset_split_seed = 12345

    load_model_folder_path: str = "./SavedModels/0.8534_2021-04-14 12:07:55.613864"  # Must set

    max_eos_predicted_length = 50
    
    # Loss function for measuring the loss on evaluation
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    batch_size: int = 32  # batch_size when running evaluation on the

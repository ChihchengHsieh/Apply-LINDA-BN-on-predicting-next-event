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
    dataset_split_seed = 12345

    # Must set
    # load_model_folder_path: str = "./SavedModels/0.8534_2021-04-14 12:07:55.613864"  # AWO
    # load_model_folder_path: str = "./SavedModels/0.8524_2021-05-04 22:56:36.630971"  # WO
    load_model_folder_path: str = "./SavedModels/0.8343_2021-05-04 23:37:50.983155"  # A

    max_eos_predicted_length = 50
    
    # Loss function for measuring the loss on evaluation
    loss: SelectableLoss = SelectableLoss.CrossEntropy
    batch_size: int = 32  # batch_size when running evaluation on the

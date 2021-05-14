from dataclasses import dataclass
from Parameters.Enums import SelectableLoss

@dataclass
class PredictingParameters(object):
    '''
    Storing the parameters for controlling the predicting.
    '''

    ######################################
    # Parameters for predicting
    ######################################
    
    load_model_folder_path: str = None
    max_eos_predicted_length:int = 50
    batch_size: int = 32  # batch_size when running evaluation on the

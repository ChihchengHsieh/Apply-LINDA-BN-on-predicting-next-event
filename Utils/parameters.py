import os 
from Parameters.TrainingParameters import TrainingParameters
from Parameters.EnviromentParameters import EnviromentParameters
import json

def load_parameters(folder_path: str) -> TrainingParameters:
    parameters_loading_path = os.path.join(
        folder_path, EnviromentParameters.parameters_save_file_name__
    )
    with open(parameters_loading_path, "r") as output_file:
        parameters = json.load(output_file)
    return TrainingParameters(**parameters)

from Controller.TrainingParameters import TrainingParameters
from Data.DataLoader import DataLoader
import os
import csv
import shutil

import pandas as pd
import numpy as np

from Utils import FileUtils


class TrainingController:
    """
    This is the man class encharged of the model training
    """

    def __init__(self):
        """constructor"""
        self.data_loader = DataLoader(
            filePath=TrainingParameters.file_path,
            append_start_end=False,
            column_names=TrainingParameters.column_names,
            filter_d_attrib=TrainingParameters.filter_d_attrib,
            one_timeStamp=TrainingParameters.one_timestamp,
            timeformat=TrainingParameters.timeformat,
            verbose=TrainingParameters.verbose
        )

        assert((not self.data_loader.log_train is None)and (not self.data_loader.log_test is None), "Data didn't load properly.")

        TrainingParameters.output = os.path.join('output_files', FileUtils.folder_id())

    

    

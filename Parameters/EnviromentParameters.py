from Parameters.Enums import PreprocessedDfType

class EnviromentParameters(object):

    #####################################
    # BPI 2012 dataset
    #####################################

    class BPI2020Dataset(object):
        file_path: str = "./datasets/event_logs/BPI_Challenge_2012.xes"
        preprocessed_foldr_path = "./datasets/preprocessed/BPI_Challenge_2012"
        preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.Pickle

    #####################################
    # Diabetes dataset
    #####################################

    class DiabetesDataset(object):
        file_path = './datasets/medical/diabetes.csv'




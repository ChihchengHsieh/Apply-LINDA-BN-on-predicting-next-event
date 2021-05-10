from Parameters.Enums import PreprocessedDfType

class EnviromentParameters(object):

    #####################################
    # BPI 2012 dataset
    #####################################

    class BPI2020Dataset(object):
        file_path: str = "./datasets/event_logs/BPI_Challenge_2012.xes"
        preprocessed_foldr_path = "./datasets/preprocessed/BPI_Challenge_2012"
        preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.Pickle

    class HelpDeskDataset(object):
        file_path: str = "./datasets/event_logs/Helpdesk.xes"
        preprocessed_foldr_path = "./datasets/preprocessed/Helpdesk"
        preprocessed_df_type: PreprocessedDfType = PreprocessedDfType.Pickle

    #####################################
    # Diabetes dataset
    #####################################

    class DiabetesDataset(object):
        file_path = './datasets/medical/diabetes.csv'
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
        target_name = "Outcome"

    class BreastCancerDataset(object):
        file_path = './datasets/medical/breast_cancer.csv'
        feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
        target_name = "diagnosis"





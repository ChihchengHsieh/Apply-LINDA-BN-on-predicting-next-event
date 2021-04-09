class TrainingParameters:
    verbose = True
    column_names: dict(str, str) = {'Case ID': 'caseid',
                                            'Activity': 'task',
                                            'lifecycle:transition': 'event_type',
                                            'Resource': 'user'}

    one_timestamp: bool = False
    timeformat: str = '%Y-%m-%dT%H:%M:%S.%f'
    model_family: str = 'gru_cx'
    opt_method: str = 'rand_hpc'  # 'rand_hpc', 'bayesian'
    max_eval: int = 1
    file_path: str = '../Data/event_logs/BPI_Challenge_2012.xes'
    filter_d_attrib: bool =  False
    output: str

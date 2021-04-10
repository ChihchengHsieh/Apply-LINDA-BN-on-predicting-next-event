class TrainingParameters:
    '''
    LSTM baseline model parameters
    '''
    embedding_dim: int = 64
    lstm_hidden:int  = 32
    dropout: float = .8
    num_lstm_layers: int = 2

    bpi_2012_path = '../Data/event_logs/BPI_Challenge_2012.xes'
    

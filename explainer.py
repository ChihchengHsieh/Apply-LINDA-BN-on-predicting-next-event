from Data.BPI2012Dataset import BPI2012Dataset
from Parameters.TrainingParameters import TrainingParameters
import math
from datetime import datetime
from IPython.core.display import display, HTML
from Controller.ExplainingController import ExplainingController
import os

def main():
    dataset = BPI2012Dataset(
                filePath=TrainingParameters.bpi_2012_path,
        preprocessed_folder_path=TrainingParameters.preprocessed_bpi_2012_folder_path,
                preprocessed_df_type=TrainingParameters.preprocessed_df_type,
            )

    index = 5
    portion = .8
    full_trace = dataset.list_of_index_to_vocab(dataset[index]["trace"])
    trace_to_predict = full_trace[ : math.ceil(len(full_trace)*portion)]
    explainer = ExplainingController()
    predicted_list, bn, inference, infoBN, markov_blanket  =  explainer.predict_next_lindaBN_explain(trace_to_predict, 5)
    html_page = explainer.generate_html_page_from_graphs(bn, inference, infoBN, markov_blanket)
    path_to_explanation = './Explanations'
    os.makedirs(path_to_explanation, exist_ok=True)
    # Save info
    save_path = os.path.join(path_to_explanation, '%s_graphs_LINDA-BN.html' % (datetime.now()))
    with open (save_path, 'w' )as output_file:
        output_file.write(html_page)


if __name__ == "__main__":
    main()
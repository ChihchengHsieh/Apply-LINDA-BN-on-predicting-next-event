import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


from Controller.PredictingController import PredictingController
import sys

# Run test with: python3 predict.py  ./PredictingObject/testing_file.json


def main(argv):
    path = argv[0]

    # Determine if using argmax or sample.
    if ("--argmax" in argv):
        use_argmax = True
    else:
        use_argmax = False

    # Determine if show loaded model info before starting.
    if ("--model-info" in argv):
        show_model_info = True
    else:
        show_model_info = False

    # Determine if only run certain steps.
    n_list = [i for i in argv if i.startswith("--n=")]
    if (len(n_list) > 0):
        n_steps = int(n_list[0].split("=")[-1])
    else:
        n_steps = None

    predictor = PredictingController()

    # Show loaded model info
    if show_model_info:
        predictor.show_model_info()

    predictor.load_json_for_predicting(
        path, n_steps=n_steps, use_argmax=use_argmax)

    input("Press any button to end the session...")


if __name__ == "__main__":
    main(sys.argv[1:])

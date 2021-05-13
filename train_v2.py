import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from  Controller import TrainingController_V2
import sys

def main(argv):
     # Determine if show loaded model info before starting.
    show_model_info = True if ("--model-info" in argv) else False

    trainer = TrainingController_V2()

    # Show loaded model arch before training 
    if show_model_info:
        trainer.show_model_info()

    trainer.train()
    trainer.save_training_result(__file__)

if __name__ == "__main__":
    main(sys.argv[1:])
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


from Controller.TrainingController import TrainingController
import sys

def main(argv):
     # Determine if show loaded model info before starting.
    if ("--model-info" in argv):
        show_model_info = True
    else:
        show_model_info = False

    trainer = TrainingController()

    # Show loaded model arch before training 
    if show_model_info:
        trainer.show_model_info()

    trainer.train()
    trainer.save_training_result(__file__)
        
    input("Press any button to end the session...")

if __name__ == "__main__":
    main(sys.argv[1:])

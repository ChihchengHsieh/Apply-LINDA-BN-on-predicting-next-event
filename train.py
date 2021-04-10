from Controller.TrainingParameters import TrainingParameters
from Controller.TrainingController import TrainingController
import sys

def main(argv):
    trainer = TrainingController(
        dataset= TrainingParameters.dataset,
        model= TrainingParameters.model,
        optimizer=TrainingParameters.optimizer,
        loss= TrainingParameters.loss,
    )
    trainer.train(TrainingParameters.stop_epoch)

if __name__ == "__main__":
    main(sys.argv[1:])
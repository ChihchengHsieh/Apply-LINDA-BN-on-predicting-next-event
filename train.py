from Controller.TrainingController import TrainingController
import sys


def main(argv):
    trainer = TrainingController()
    trainer.train()

if __name__ == "__main__":
    main(sys.argv[1:])

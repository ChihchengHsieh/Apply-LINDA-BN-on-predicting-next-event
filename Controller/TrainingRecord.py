import matplotlib.pyplot as plt
import json


class TrainingRecord:
    records_save_file_name = "records.json"
    figure_save_file_name = "plot.png"
    def __init__(self, record_freq_in_step: int):
        plt.ion()

        self.train_accuracy_records: list[float] = []
        self.train_loss_records: list[float] = []
        self.validation_acuuracy_records: list[float] = []
        self.validation_loss_records: list[float] = []
        self.record_freq_in_step: int = record_freq_in_step
        self.fig = plt.figure(figsize=(20, 10), dpi=80)

    def record_training_info(self, train_accuracy: float, train_loss: float, validation_accuracy: float, validation_loss: float, ):
        self.train_accuracy_records.append(train_accuracy)
        self.train_loss_records.append(train_loss)
        self.validation_acuuracy_records.append(validation_accuracy)
        self.validation_loss_records.append(validation_loss)

    def plot_records(self):
        if not plt.fignum_exists(self.fig.number):
            plt.show()

        plt.subplot().cla()

        # Plot loss
        plt.subplot(211)
        self.plot_loss()

        # Plot accuracy
        plt.subplot(212)
        self.plot_accuracy()

        plt.tight_layout()

    def plot_loss(self):
        plt.plot(self.train_loss_records, marker='o', label='Training loss')
        plt.plot(self.validation_loss_records,
                 marker='o', label='Validation loss')
        plt.ylabel('Loss', fontsize=8)
        plt.xlabel('Every %d steps' % (self.record_freq_in_step))
        plt.legend(loc='upper left')
        plt.draw()
        plt.pause(0.001)

    def plot_accuracy(self):
        plt.plot(self.train_accuracy_records,
                 marker='o', label='Training accuracy')
        plt.plot(self.validation_acuuracy_records,
                 marker='o', label='Validation accuracy')
        plt.ylabel('Acuuracy', fontsize=8)
        plt.xlabel('Every %d steps' % (self.record_freq_in_step))
        plt.legend(loc='upper left')
        plt.draw()
        plt.pause(0.001)

    def save_records_to_file(self, path: str):
        all_records = {
            "train_accuracy_records": self.train_accuracy_records,
            "train_loss_records": self.train_loss_records,
            "validation_acuuracy_records": self.validation_acuuracy_records,
            "validation_loss_records": self.validation_loss_records,
        }

        with open(path, 'w') as output_file:
            json.dump(all_records, output_file, indent='\t')

    def load_records(self, path: str):
        with open(path, 'r') as output_file:
            all_records: dict[str, list] = json.load(output_file)
        self.train_accuracy_records = all_records["train_accuracy_records"]
        self.train_loss_records = all_records["train_loss_records"]
        self.validation_acuuracy_records = all_records["validation_acuuracy_records"]
        self.validation_loss_records = all_records["validation_loss_records"]

    def save_figure(self, path: str, dpi: int = 80):
        if not self.fig is None:
            plt.savefig(path, dpi=dpi)

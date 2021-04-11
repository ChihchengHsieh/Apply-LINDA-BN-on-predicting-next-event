import matplotlib.pyplot as plt


class TrainingRecord:
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

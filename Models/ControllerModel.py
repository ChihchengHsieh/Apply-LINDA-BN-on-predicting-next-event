from abc import ABC
import abc

class ControllerModel(abc.ABC):

    @abc.abstractclassmethod
    def data_forward(self, data):
        pass

    @abc.abstractclassmethod
    def get_accuracy(self, out, target):
        pass

    @abc.abstractclassmethod
    def get_loss(self, loss_fn: callable, out, target):
        pass

    @abc.abstractclassmethod
    def get_prediction_list_from_out(self, out,  mask=None):
        pass

    @abc.abstractclassmethod
    def get_target_list_from_target(self, target,  mask=None):
        pass

    @abc.abstractclassmethod
    def generate_mask(self, target):
        pass

    @abc.abstractclassmethod
    def get_labels(self):
        pass
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
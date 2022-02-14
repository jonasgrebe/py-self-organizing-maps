from abc import ABC as Abstract
from abc import abstractmethod
import numpy as np

class Topology(Abstract):

    @abstractmethod
    def get_node(self, node_idx):
        pass

    @abstractmethod
    def get_neighbors_of_node(self, node_idx, radius):
        pass

    @abstractmethod
    def metric(self, x, y):
        pass

    @abstractmethod
    def get_number_of_nodes(self):
        pass

    def __len__(self):
        return self.get_number_of_nodes()

    @abstractmethod
    def plot_map(self, weights, axis=None, title=""):
        pass

    @abstractmethod
    def plot_nodes(self, weights, axis=None, title=""):
        pass

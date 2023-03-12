from abc import ABCMeta, abstractmethod
from torchvision.datasets.vision import VisionDataset

from superpoint.utils.tools import dict_update


class BaseDataset(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the following methods:
        `_init_dataset` and `_get_data`.
    Additionally, the following static attributes should be defined:
        default_config: A dictionary of potential default configuration values (e.g. the
            size of the validation set).
    """
    split_names = ['training', 'validation', 'test']

    
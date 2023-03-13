from abc import ABCMeta, abstractmethod
from torchvision.datasets.vision import VisionDataset

from superpoint.utils.tools import dict_update

from typing import Callable, Optional, Any
from PIL import Image

class BaseDataset(VisionDataset):
    """Base model class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the following methods:
        `_init_dataset` and `_get_data`.
    Additionally, the following static attributes should be defined:
        default_config: A dictionary of potential default configuration values (e.g. the
            size of the validation set).
    """
    # def __init__(self, root: str, train: bool, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
    #     super(BaseDataset, self).__init__(root, transform=transform, target_transform=target_transform)
    #     raise NotImplementedError
    @abstractmethod
    def _init_dataset(self, **config):
        """Prepare the dataset for reading.

        This method should configure the dataset for later fetching through `_get_data`,
        such as downloading the data if it is not stored locally, or reading the list of
        data files from disk. Ideally, especially in the case of large images, this
        method shoudl NOT read all the dataset into memory, but rather prepare for faster
        seubsequent fetching.

        Arguments:
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            An object subsequently passed to `_get_data`, e.g. a list of file paths and
            set splits.
        """
        raise NotImplementedError

    # make read init param from yaml config
    def __init__(self, **config):
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self.root = self.config['root']
        self.train = self.config['train']
        self.transform = self.config['transform']
        self.target_transform = self.config['target_transform']
        self.dataset = self._init_dataset(**self.config)
    
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
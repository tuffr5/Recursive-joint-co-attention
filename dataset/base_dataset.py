"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
from abc import ABC, abstractmethod
import h5py
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        if opt.phase == 'train':
            self.opt.dir_order = opt.dir_order_train
        elif opt.phase == 'val':
            self.opt.dir_order = opt.dir_order_val
        elif opt.phase == 'test':
            self.opt.dir_order = opt.dir_order_test
        else:
            raise NotImplementedError('dataset phase [%s] is not found' % opt.phase)

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.order_dir)

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def make_dataset(video_dir, audio_dir, label_dir, order_dir):
    with h5py.File(order_dir, 'r') as hf:
        orders = hf['order'][:]
    with h5py.File(audio_dir, 'r') as hf:
        audio_features = hf['avadataset'][:]
    with h5py.File(label_dir, 'r') as hf:
        labels = hf['avadataset'][:]
    with h5py.File(video_dir, 'r') as hf:
        video_features = hf['avadataset'][:]

    return video_features, audio_features, labels, orders

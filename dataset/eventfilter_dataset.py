from dataset.base_dataset import BaseDataset, make_dataset
import torch
import numpy as np


class EventfilterDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.videos, self.audios, self.labels, self.orders = make_dataset(self.opt.dir_video, self.opt.dir_audio, self.opt.dir_labels, self.opt.dir_order)

    def __getitem__(self, index):

        video = self.videos[self.orders[index]]
        audio = self.audios[self.orders[index]]
        label = self.labels[self.orders[index]]
        event_cov = np.zeros((10, 10))
        e_label = np.zeros(10)
        for i in range(10):
            e_label[i] = 0 if np.argmax(label[i, :]) == 28 else 1

        for i in range(10):
            for j in range(10):
                if np.argmax(label[i, :]) == 28:
                    event_cov[i, :] = 0
                else:
                    if np.argmax(label[j, :]) != 28:
                        event_cov[i, j] = 1

        np.fill_diagonal(event_cov, 1)

        return {'audio': torch.from_numpy(audio).float(), 'video': torch.from_numpy(video).float(),
                'label': torch.from_numpy(e_label).float(), 'event_cov': torch.from_numpy(event_cov).float()}
        # return {'label': e_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.orders)

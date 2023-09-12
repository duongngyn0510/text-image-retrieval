import clip
import numpy as np
from torch.utils.data import Dataset, BatchSampler
from utils import preprocess


class FashionDataset(Dataset):
    """_summary_
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, images_list, prompt_list, label_list, preprocess):
        self.images_list = images_list
        self.prompt_list = prompt_list
        self.label_list = label_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]
        image_tensor = self.preprocess(image)

        prompt = self.prompt_list[idx]
        prompt_token = clip.tokenize([prompt])[0]

        label = self.label_list[idx]
        return image_tensor, prompt_token, label
    

class BalancedBatchSampler(BatchSampler):
    """
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    

class FashionInfer(Dataset):
    def __init__(self, images_list):
        self.images_list = images_list
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        image_tensor = preprocess()(self.images_list[idx])
        return image_tensor
    
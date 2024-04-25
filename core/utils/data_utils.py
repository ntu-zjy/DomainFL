import copy
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np

# def build_subset(dataObject, num_classes, batch_size):
#     # Ensure num_classes is not greater than available classes
#     num_classes = min(num_classes, len(dataObject.train_dataset.class_to_idx))

#     # Collect indices of data samples belonging to the specified number of classes
#     selected_indices = [i for i, (_, label) in enumerate(dataObject.train_dataset) if label < num_classes]

#     # Create subset using these indices
#     subset = Subset(dataObject.train_dataset, selected_indices)

#     # Create a DataLoader for the subset
#     subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

#     return subset_loader

class CustomSubset(Subset):
    def __init__(self, dataset, indices, class_to_idx):
        super().__init__(dataset, indices)
        self.class_to_idx = class_to_idx

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, class_to_idx):
        super().__init__(datasets)
        self.class_to_idx = class_to_idx

def build_subset(dataObject, num_classes):
    # Use set for faster lookup
    allowed_labels = set(range(num_classes))
    valid_class_labels = dataObject.train_dataset.img_labels[dataObject.train_dataset.img_labels['label'].isin(allowed_labels)]
    new_class_to_idx = dict(zip(valid_class_labels['class_name'], valid_class_labels['label']))

    # Filter indices for both training and testing datasets
    train_labels = dataObject.train_dataset.img_labels['label'].values
    test_labels = dataObject.test_dataset.img_labels['label'].values

    train_indices = np.where(train_labels < num_classes)[0]
    test_indices = np.where(test_labels < num_classes)[0]

    # Use CustomSubset to create new subsets
    dataObject.train_dataset = CustomSubset(dataObject.train_dataset, train_indices, new_class_to_idx)
    dataObject.test_dataset = CustomSubset(dataObject.test_dataset, test_indices, new_class_to_idx)

    # Update DataLoader and classnames
    dataObject.train_loader = DataLoader(dataObject.train_dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True)
    dataObject.test_loader = DataLoader(dataObject.test_dataset, batch_size=dataObject.test_loader.batch_size, num_workers=dataObject.test_loader.num_workers, pin_memory=True)
    dataObject.update_classnames()

    return dataObject

def concat_datasets(dataObjects):

    # Concatenate train datasets
    train_datasets = [dataObject.train_dataset for dataObject in dataObjects]
    class_to_idx = train_datasets[0].class_to_idx
    train_dataset = CustomConcatDataset(train_datasets, class_to_idx)

    # Concatenate test datasets
    test_datasets = [dataObject.test_dataset for dataObject in dataObjects]
    test_dataset = CustomConcatDataset(test_datasets, class_to_idx)

    # Create a new DataLoader for the concatenated train dataset
    batch_size = dataObjects[0].train_loader.batch_size
    num_workers = dataObjects[0].train_loader.num_workers
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataLoader for the concatenated test dataset
    batch_size = dataObjects[0].test_loader.batch_size
    num_workers = dataObjects[0].test_loader.num_workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


    # Create a new DataObject with the concatenated datasets
    dataObject = copy.deepcopy(dataObjects[0])
    dataObject.train_dataset = train_dataset
    dataObject.test_dataset = test_dataset
    dataObject.train_loader = train_loader
    dataObject.test_loader = test_loader
    dataObject.update_classnames()

    return dataObject
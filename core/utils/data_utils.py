import copy
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import math

class CustomSubset(Subset):
    def __init__(self, dataset, indices, class_to_idx, image_labels=None):
        super().__init__(dataset, indices)
        self.class_to_idx = class_to_idx
        self.image_labels = image_labels

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets, class_to_idx):
        super().__init__(datasets)
        self.class_to_idx = class_to_idx

def build_subset(dataObject, num_classes):
    # Use set for faster lookup
    allowed_labels = set(range(num_classes))
    # valid_class_labels = dataObject.train_dataset.img_labels[dataObject.train_dataset.img_labels['label'].isin(allowed_labels)]
    # new_class_to_idx = dict(zip(valid_class_labels['class_name'], valid_class_labels['label']))

    # only keep the num_classes of classes in the class_to_idx
    new_class_to_idx = {k: v for k, v in dataObject.train_dataset.class_to_idx.items() if v < num_classes}

    # Filter indices for both training and testing datasets
    train_labels = dataObject.train_dataset.img_labels['label'].values
    test_labels = dataObject.test_dataset.img_labels['label'].values

    train_indices = np.where(train_labels < num_classes)[0]
    test_indices = np.where(test_labels < num_classes)[0]

    train_labels = train_labels[train_indices]
    test_labels = test_labels[test_indices]

    # Use CustomSubset to create new subsets
    dataObject.train_dataset = CustomSubset(dataObject.train_dataset, train_indices, new_class_to_idx, train_labels)
    dataObject.test_dataset = CustomSubset(dataObject.test_dataset, test_indices, new_class_to_idx, test_labels)
    # Update DataLoader and classnames
    dataObject.train_loader = DataLoader(dataObject.train_dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True)
    dataObject.test_loader = DataLoader(dataObject.test_dataset, batch_size=dataObject.test_loader.batch_size, num_workers=dataObject.test_loader.num_workers, pin_memory=True)
    dataObject.update_classnames()

    return dataObject


def build_subset_mixed(dataObject, num_classes, ratios):
    import numpy as np
    from collections import defaultdict

    # Use set for faster lookup
    allowed_labels = set(range(num_classes))
    new_class_to_idx = {k: v for k, v in dataObject.train_dataset.class_to_idx.items() if v < num_classes}

    # Filter indices for both training and testing datasets
    train_labels = dataObject.train_dataset.img_labels['label'].values
    test_labels = dataObject.test_dataset.img_labels['label'].values

    train_indices = np.where(train_labels < num_classes)[0]
    test_indices = np.where(test_labels < num_classes)[0]

    train_labels = train_labels[train_indices]
    test_labels = test_labels[test_indices]

    # Create subsets based on ratios
    def create_subsets(indices, labels, ratios):
        subsets = defaultdict(list)
        for class_label in range(num_classes):
            class_indices = indices[labels == class_label]
            num_samples = len(class_indices)
            num_per_class_subset = math.ceil(num_samples * ratios[0])
            # print(f"Class {class_label}: {num_samples} samples")  # Debugging line

            start = 0
            for i, _ in enumerate(ratios):
                end = (i+1) * num_per_class_subset
                subsets[i].extend(class_indices[start:end].tolist())
                start = end
            subsets[len(ratios)].extend(class_indices[-num_per_class_subset:].tolist())

        # for i in range(len(ratios) + 1):
        #     print(f"Subset {i}: {len(subsets[i])} samples")  # Debugging line

        return [np.array(subsets[i], dtype=int) for i in range(len(ratios) + 1)]

    # Create train and test subsets
    train_subsets = create_subsets(train_indices, train_labels, ratios)
    test_subsets = create_subsets(test_indices, test_labels, ratios)
    # print(train_subsets)
    # print(test_subsets)
    # Create data loaders for each subset
    train_datasets = [CustomSubset(dataObject.train_dataset, subset_indices, new_class_to_idx, train_labels[subset_indices]) for subset_indices in train_subsets]
    test_datasets = [CustomSubset(dataObject.test_dataset, subset_indices, new_class_to_idx, test_labels[subset_indices]) for subset_indices in test_subsets]

    train_loaders = [DataLoader(dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True) for dataset in train_datasets]
    test_loaders = [DataLoader(dataset, batch_size=dataObject.test_loader.batch_size, num_workers=dataObject.test_loader.num_workers, pin_memory=True) for dataset in test_datasets]

    # Return a list of data objects with updated datasets and loaders
    data_objects = []
    for i in range(len(ratios) + 1):
        new_data_object = copy.deepcopy(dataObject)  # Assuming dataObject has a copy method
        new_data_object.train_dataset = train_datasets[i]
        new_data_object.test_dataset = test_datasets[i]
        new_data_object.train_loader = train_loaders[i]
        new_data_object.test_loader = test_loaders[i]
        new_data_object.update_classnames()
        data_objects.append(new_data_object)

    return data_objects

def build_subset_mixed_past(dataObject, num_classes, ratios):
    import numpy as np
    from collections import defaultdict

    # Use set for faster lookup
    allowed_labels = set(range(num_classes))
    new_class_to_idx = {k: v for k, v in dataObject.train_dataset.class_to_idx.items() if v < num_classes}

    # Filter indices for both training and testing datasets
    train_labels = dataObject.train_dataset.img_labels['label'].values
    test_labels = dataObject.test_dataset.img_labels['label'].values

    train_indices = np.where(train_labels < num_classes)[0]
    test_indices = np.where(test_labels < num_classes)[0]

    train_labels = train_labels[train_indices]
    test_labels = test_labels[test_indices]

    # Create subsets based on ratios
    def create_subsets(indices, labels, ratios):
        # print("len(indices):", len(indices))
        # print("ratios:", ratios)
        subsets = defaultdict(list)
        for class_label in range(num_classes):
            class_indices = indices[labels == class_label]
            num_samples = len(class_indices)
            # print(f"Class {class_label}: {num_samples} samples")  # Debugging line

            start = 0
            for i, ratio in enumerate(ratios):
                end = start + int(ratio * num_samples)
                if end > start:  # Ensure there are samples in this range
                    subsets[i].extend(class_indices[start:end].tolist())
                start = end
            # Add remaining samples to the last subset
            if start < num_samples:
                subsets[len(ratios)].extend(class_indices[start:].tolist())

        # for i in range(len(ratios) + 1):
        #     print(f"Subset {i}: {len(subsets[i])} samples")  # Debugging line

        return [np.array(subsets[i], dtype=int) for i in range(len(ratios) + 1)]

    # Create train and test subsets
    train_subsets = create_subsets(train_indices, train_labels, ratios)
    test_subsets = create_subsets(test_indices, test_labels, ratios)
    # print(train_subsets)
    # print(test_subsets)
    # Create data loaders for each subset
    train_datasets = [CustomSubset(dataObject.train_dataset, subset_indices, new_class_to_idx, train_labels[subset_indices]) for subset_indices in train_subsets]
    test_datasets = [CustomSubset(dataObject.test_dataset, subset_indices, new_class_to_idx, test_labels[subset_indices]) for subset_indices in test_subsets]

    train_loaders = [DataLoader(dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True) for dataset in train_datasets]
    test_loaders = [DataLoader(dataset, batch_size=dataObject.test_loader.batch_size, num_workers=dataObject.test_loader.num_workers, pin_memory=True) for dataset in test_datasets]

    # Return a list of data objects with updated datasets and loaders
    data_objects = []
    for i in range(len(ratios) + 1):
        new_data_object = copy.deepcopy(dataObject)  # Assuming dataObject has a copy method
        new_data_object.train_dataset = train_datasets[i]
        new_data_object.test_dataset = test_datasets[i]
        new_data_object.train_loader = train_loaders[i]
        new_data_object.test_loader = test_loaders[i]
        new_data_object.update_classnames()
        data_objects.append(new_data_object)

    return data_objects

def build_subset_mixed_2(dataObject, num_classes, ratios):
    import numpy as np
    from collections import defaultdict

    # Use set for faster lookup
    allowed_labels = set(range(num_classes))
    new_class_to_idx = {k: v for k, v in dataObject.train_dataset.class_to_idx.items() if v < num_classes}

    # Filter indices for both training and testing datasets
    train_labels = dataObject.train_dataset.img_labels['label'].values
    test_labels = dataObject.test_dataset.img_labels['label'].values

    train_indices = np.where(train_labels < num_classes)[0]
    test_indices = np.where(test_labels < num_classes)[0]

    train_labels = train_labels[train_indices]
    test_labels = test_labels[test_indices]

    # Create subsets based on ratios
    def create_subsets(indices, labels, ratios):
        print("len(indices):", len(indices))
        print("ratios:", ratios)
        subsets = defaultdict(list)
        for class_label in range(num_classes):
            class_indices = indices[labels == class_label]
            num_samples = len(class_indices)
            start = 0
            for i, ratio in enumerate(ratios):
                end = start + int(ratio * num_samples)
                subsets[i].extend(class_indices[start:end])
                start = end
        return [np.array(subsets[i], dtype=int) for i in range(len(ratios) + 1)]

    print("len(train_indices):", len(train_indices))
    print("len(test_indices):", len(test_indices))
    # Create train and test subsets
    train_subsets = create_subsets(train_indices, train_labels, ratios)
    test_subsets = create_subsets(test_indices, test_labels, ratios)

    # Create data loaders for each subset
    train_datasets = [CustomSubset(dataObject.train_dataset, subset_indices, new_class_to_idx, train_labels[subset_indices]) for subset_indices in train_subsets]
    test_datasets = [CustomSubset(dataObject.test_dataset, subset_indices, new_class_to_idx, test_labels[subset_indices]) for subset_indices in test_subsets]

    train_loaders = [DataLoader(dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True) for dataset in train_datasets]
    test_loaders = [DataLoader(dataset, batch_size=dataObject.test_loader.batch_size, num_workers=dataObject.test_loader.num_workers, pin_memory=True) for dataset in test_datasets]

    # Return a list of data objects with updated datasets and loaders
    data_objects = []
    for i in range(len(ratios) + 1):
        new_data_object = dataObject.copy()  # Assuming dataObject has a copy method
        new_data_object.train_dataset = train_datasets[i]
        new_data_object.test_dataset = test_datasets[i]
        new_data_object.train_loader = train_loaders[i]
        new_data_object.test_loader = test_loaders[i]
        new_data_object.update_classnames()
        data_objects.append(new_data_object)

    return data_objects

def split_train_and_val(dataObject, val_percent=0.2):
    targets = dataObject.train_dataset.image_labels
    print("len(targets):", len(targets))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percent, random_state=1)

    for train_index, val_index in sss.split(list(range(len(targets))), targets):
        pass
    # print("train_index:", train_index)
    # print("val_index:", val_index)
    train_labels = targets[train_index]
    val_labels = targets[val_index]
    # print("train_labels:", train_labels)

    class_to_idx = dataObject.train_dataset.class_to_idx
    train_dataset = copy.deepcopy(dataObject.train_dataset)
    # Use CustomSubset to create new subsets
    dataObject.train_dataset = CustomSubset(train_dataset, train_index, class_to_idx, train_labels)
    dataObject.val_dataset = CustomSubset(train_dataset, val_index, class_to_idx, val_labels)
    # Update DataLoader and classnames
    dataObject.train_loader = DataLoader(dataObject.train_dataset, shuffle=True, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True)
    dataObject.val_loader = DataLoader(dataObject.val_dataset, batch_size=dataObject.train_loader.batch_size, num_workers=dataObject.train_loader.num_workers, pin_memory=True)

    return dataObject

def local_adaptation_subset_trainloader(train_dataset, train_loader, num_percent):
    class_to_idx = train_dataset.class_to_idx
    total_train_samples = len(train_dataset)
    num_samples = int(total_train_samples * num_percent / 100)

    selected_indices = np.random.choice(total_train_samples, num_samples, replace=False)

    train_dataset = CustomSubset(train_dataset, selected_indices, class_to_idx)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_loader.batch_size, num_workers=train_loader.num_workers, pin_memory=True)

    return train_loader

def concat_datasets(dataObjects):
    # TODO:

    # Concatenate train datasets
    train_datasets = [dataObject.train_dataset for dataObject in dataObjects]
    class_to_idx = train_datasets[0].class_to_idx
    train_dataset = CustomConcatDataset(train_datasets, class_to_idx)

    # Concatenate test datasets
    test_datasets = [dataObject.test_dataset for dataObject in dataObjects]
    test_dataset = CustomConcatDataset(test_datasets, class_to_idx)

    # Concatenate val datasets
    val_datasets = [dataObject.val_dataset for dataObject in dataObjects]
    val_dataset = CustomConcatDataset(val_datasets, class_to_idx)

    # Create a new DataLoader for the concatenated train dataset
    batch_size = dataObjects[0].train_loader.batch_size
    num_workers = dataObjects[0].train_loader.num_workers
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataLoader for the concatenated test dataset
    batch_size = dataObjects[0].test_loader.batch_size
    num_workers = dataObjects[0].test_loader.num_workers
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataLoader for the concatenated val dataset
    batch_size = dataObjects[0].val_loader.batch_size
    num_workers = dataObjects[0].val_loader.num_workers
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Create a new DataObject with the concatenated datasets
    dataObject = copy.deepcopy(dataObjects[0])
    dataObject.train_dataset = train_dataset
    dataObject.test_dataset = test_dataset
    dataObject.train_loader = train_loader
    dataObject.test_loader = test_loader
    dataObject.val_dataset = val_dataset
    dataObject.val_loader = val_loader
    dataObject.update_classnames()

    return dataObject

def concat_test_datasets(clientObjects):
    # Concatenate test datasets
    train_datasets = [clientObject.train_dataset for clientObject in clientObjects]
    class_to_idx = train_datasets[0].class_to_idx
    test_datasets = [clientObject.test_dataset for clientObject in clientObjects]
    test_dataset = CustomConcatDataset(test_datasets, class_to_idx)

    # Create a new DataLoader for the concatenated test dataset
    batch_size = clientObjects[0].test_dataloader.batch_size
    num_workers = clientObjects[0].test_dataloader.num_workers
    concated_test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return concated_test_loader

def generate_domain_label(cid, dataObjects):
    num_test_datasets = [len(dataObject.test_dataset) for dataObject in dataObjects]
    print("num_test_datasets:", num_test_datasets)
    # make the domain label for the test dataset
    domain_label = np.zeros(sum(num_test_datasets))
    domain_range = num_test_datasets[cid]
    start_idx = sum(num_test_datasets[:cid])
    end_idx = start_idx + domain_range
    domain_label[start_idx:end_idx] = 1
    return domain_label

def split_dataloader_by_labels(dataloader, predicted_labels, true_labels):
    # 初始化索引列表
    tp_indices = []
    tn_indices = []
    fp_indices = []
    fn_indices = []

    for index, (pred_label, true_label) in enumerate(zip(predicted_labels, true_labels)):
        if pred_label == true_label == 1:
            tp_indices.append(index)
        elif pred_label == true_label == 0:
            tn_indices.append(index)
        elif pred_label == 1 and true_label == 0:
            fp_indices.append(index)
        elif pred_label == 0 and true_label == 1:
            fn_indices.append(index)

    tp_dataset = Subset(dataloader.dataset, tp_indices)
    tn_dataset = Subset(dataloader.dataset, tn_indices)
    fp_dataset = Subset(dataloader.dataset, fp_indices)
    fn_dataset = Subset(dataloader.dataset, fn_indices)

    tp_dataloader = DataLoader(tp_dataset, batch_size=dataloader.batch_size, shuffle=False)
    tn_dataloader = DataLoader(tn_dataset, batch_size=dataloader.batch_size, shuffle=False)
    fp_dataloader = DataLoader(fp_dataset, batch_size=dataloader.batch_size, shuffle=False)
    fn_dataloader = DataLoader(fn_dataset, batch_size=dataloader.batch_size, shuffle=False)

    return tp_dataloader, tn_dataloader, fp_dataloader, fn_dataloader
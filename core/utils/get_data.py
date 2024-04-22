import numpy as np
import random

from data1.datasets_utils.cars import Cars
from data1.datasets_utils.cifar10 import CIFAR10
from data1.datasets_utils.cifar100 import CIFAR100
from data1.datasets_utils.dtd import DTD
from data1.datasets_utils.flowers102 import Flowers102
from data1.datasets_utils.food101 import Food101
from data1.datasets_utils.imagenette import Imagenette
from data1.datasets_utils.mnist import MNIST
from data1.datasets_utils.fashionmnist import FashionMNIST


data1 = ['Flowers102', 'Food101', 'MNIST', 'Imagenette', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'DTD']

from data2.datasets_utils.clipart import Clipart
from data2.datasets_utils.painting import Painting
from data2.datasets_utils.real import Real
from data2.datasets_utils.sketch import Sketch
from data2.datasets_utils.quickdraw import Quickdraw
from data2.datasets_utils.infograph import Infograph

data2 = ['Clipart', 'Painting', 'Real', 'Sketch', 'Quickdraw', 'Infograph']
random.seed(1)
np.random.seed(1)

def get_data(data_name, train_preprocess, val_preprocess, root, batch_size, num_workers):
    return globals()[data_name](
        train_preprocess=train_preprocess,
        val_preprocess=val_preprocess,
        location=root,
        batch_size=batch_size,
        num_workers=num_workers)
import numpy as np
import random

from domainnet.datasets_utils.clipart import Clipart
from domainnet.datasets_utils.painting import Painting
from domainnet.datasets_utils.real import Real
from domainnet.datasets_utils.sketch import Sketch
from domainnet.datasets_utils.quickdraw import Quickdraw
from domainnet.datasets_utils.infograph import Infograph


domainnet = ['Clipart', 'Painting', 'Real', 'Sketch', 'Quickdraw', 'Infograph'] # domainnet
random.seed(1)
np.random.seed(1)

def get_data(data_name, train_preprocess, val_preprocess, batch_size, num_workers):
    return globals()[data_name](
        train_preprocess=train_preprocess,
        val_preprocess=val_preprocess,
        batch_size=batch_size,
        num_workers=num_workers)
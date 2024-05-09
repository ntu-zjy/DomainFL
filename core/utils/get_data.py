import numpy as np
import random

from data.datasets_utils.clipart import Clipart
from data.datasets_utils.painting import Painting
from data.datasets_utils.real import Real
from data.datasets_utils.sketch import Sketch
from data.datasets_utils.quickdraw import Quickdraw
from data.datasets_utils.infograph import Infograph

from office.datasets_utils.officeclipart import OfficeClipart
from office.datasets_utils.officeart import OfficeArt
from office.datasets_utils.officereal import OfficeReal
from office.datasets_utils.officeproduct import OfficeProduct

data = ['Clipart', 'Painting', 'Real', 'Sketch', 'Quickdraw', 'Infograph'] # domainnet
office = ['OfficeClipart', 'OfficeProduct', 'OfficeReal', 'OfficeArt'] # officehome
random.seed(1)
np.random.seed(1)

def get_data(data_name, train_preprocess, val_preprocess, root, batch_size, num_workers):
    return globals()[data_name](
        train_preprocess=train_preprocess,
        val_preprocess=val_preprocess,
        location=root,
        batch_size=batch_size,
        num_workers=num_workers)
import numpy as np
import random

from domainnet_zeroshot.datasets_utils.clipart import Clipart
from domainnet_zeroshot.datasets_utils.painting import Painting
from domainnet_zeroshot.datasets_utils.real import Real
from domainnet_zeroshot.datasets_utils.sketch import Sketch
from domainnet_zeroshot.datasets_utils.quickdraw import Quickdraw
from domainnet_zeroshot.datasets_utils.infograph import Infograph

from adaptiope.datasets_utils.product import product
from adaptiope.datasets_utils.reallife import reallife
from adaptiope.datasets_utils.synthetic import synthetic

from PACS.art_painting import ArtPainting
from PACS.cartoon import Cartoon
from PACS.photo import Photo
from PACS.sketchd import SketchD


domainnet = ['Clipart', 'Painting', 'Real', 'Sketch', 'Quickdraw', 'Infograph'] # domainnet
adaptiope = ['product', 'reallife', 'synthetic'] # adaptiope
PACS = ['Photo', 'Sketch', 'ArtPainting', 'Cartoon']
random.seed(1)
np.random.seed(1)

def get_data(data_name, train_preprocess, val_preprocess, batch_size, num_workers):
    return globals()[data_name](
        train_preprocess=train_preprocess,
        val_preprocess=val_preprocess,
        batch_size=batch_size,
        num_workers=num_workers)
import os
import torch
from torchvision.datasets import Flowers102 as PyTorchFlowers102

class Flowers102:
    def __init__(self,
                 train_preprocess,
                 val_preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):

        self.train_dataset = PyTorchFlowers102(
            root=location, download=True, split='train', transform=train_preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
        )

        self.test_dataset = PyTorchFlowers102(
            root=location, download=True, split='test', transform=val_preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=min(batch_size*8, 1024), shuffle=False, num_workers=num_workers
        )

        self.classnames = ['pink primrose',
                        'hard-leaved pocket orchid',
                        'canterbury bells',
                        'sweet pea',
                        'english marigold',
                        'tiger lily',
                        'moon orchid',
                        'bird of paradise',
                        'monkshood',
                        'globe thistle',
                        'snapdragon',
                        "colt's foot",
                        'king protea',
                        'spear thistle',
                        'yellow iris',
                        'globe-flower',
                        'purple coneflower',
                        'peruvian lily',
                        'balloon flower',
                        'giant white arum lily',
                        'fire lily',
                        'pincushion flower',
                        'fritillary',
                        'red ginger',
                        'grape hyacinth',
                        'corn poppy',
                        'prince of wales feathers',
                        'stemless gentian',
                        'artichoke',
                        'sweet william',
                        'carnation',
                        'garden phlox',
                        'love in the mist',
                        'mexican aster',
                        'alpine sea holly',
                        'ruby-lipped cattleya',
                        'cape flower',
                        'great masterwort',
                        'siam tulip',
                        'lenten rose',
                        'barbeton daisy',
                        'daffodil',
                        'sword lily',
                        'poinsettia',
                        'bolero deep blue',
                        'wallflower',
                        'marigold',
                        'buttercup',
                        'oxeye daisy',
                        'common dandelion',
                        'petunia',
                        'wild pansy',
                        'primula',
                        'sunflower',
                        'pelargonium',
                        'bishop of llandaff',
                        'gaura',
                        'geranium',
                        'orange dahlia',
                        'pink-yellow dahlia',
                        'cautleya spicata',
                        'japanese anemone',
                        'black-eyed susan',
                        'silverbush',
                        'californian poppy',
                        'osteospermum',
                        'spring crocus',
                        'bearded iris',
                        'windflower',
                        'tree poppy',
                        'gazania',
                        'azalea',
                        'water lily',
                        'rose',
                        'thorn apple',
                        'morning glory',
                        'passion flower',
                        'lotus lotus',
                        'toad lily',
                        'anthurium',
                        'frangipani',
                        'clematis',
                        'hibiscus',
                        'columbine',
                        'desert-rose',
                        'tree mallow',
                        'magnolia',
                        'cyclamen',
                        'watercress',
                        'canna lily',
                        'hippeastrum',
                        'bee balm',
                        'ball moss',
                        'foxglove',
                        'bougainvillea',
                        'camellia',
                        'mallow',
                        'mexican petunia',
                        'bromelia',
                        'blanket flower',
                        'trumpet creeper',
                        'blackberry lily']
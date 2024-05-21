import torch
import copy
import open_clip
import torch.nn as nn

# https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv
d = {'RN50':'openai',
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'laion2b_s32b_b82k',
    'convnext_base': 'laion400m_s13b_b51k'}

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, bias=False):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=bias),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# define a clip image classifier
class ImageEncoder(torch.nn.Module):
    def __init__(self, args, zeroshot=False):
        super().__init__()
        self.args = args
        name = args.image_encoder_name
        pretrained = d[name]
        self.zeroshot = zeroshot

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        # use the output dim of the visual encoder as the input dim of the adapter
        if 'ViT' in name or 'RN50' in name:
            self.output_dim = self.model.visual.output_dim
        else:
            self.output_dim = self.model.visual.head.proj.out_features

        self.adapter = Adapter(self.output_dim, 4, bias=False).to(args.device)
        self.global_adapter = copy.deepcopy(Adapter(self.output_dim, 4, bias=False).to(args.device))
        self.local_adapter = copy.deepcopy(Adapter(self.output_dim, 4, bias=False).to(args.device))

    def forward(self, images):
        image_features = self.model.encode_image(images)
        if self.zeroshot:
            return image_features
        else:
            return self.adapter(image_features) + image_features

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch.save(filename)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch.save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch.load(filename, map_location="cpu")


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.base = image_encoder
        self.head = classification_head

    def freeze_head(self):
        self.head.weight.requires_grad_(False)
        self.head.bias.requires_grad_(False)

    def freeze_encoder(self):
        for param in self.base.parameters():
            param.requires_grad_(False)

    def freeze_except_adapter(self):
        for name, param in self.base.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

    def forward(self, inputs):
        features = self.base(inputs)
        outputs = self.head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch.save(filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch.load(filename)

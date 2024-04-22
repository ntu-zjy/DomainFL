import torch
import open_clip
import torch.nn as nn

d = {'RN50':'openai',
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'laion2b_s32b_b82k'}

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# define a clip image classifier
class ImageEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        name = args.image_encoder_name
        pretrained = d[name]

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.adapter = Adapter(self.model.visual.output_dim, 4).to(args.device)
        self.global_adapter = Adapter(self.model.visual.output_dim, 4).to(args.device)
        self.adapter_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.adapter_beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, images):
        assert self.model is not None
        image_features = self.model.encode_image(images)
        return self.adapter_beta * self.global_adapter(image_features) + \
            self.adapter_alpha * self.adapter(image_features) + \
                (1-self.adapter_alpha-self.adapter_beta) * image_features

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch.save(filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    # @classmethod
    # def load_from_state_dict(cls, model_name, state_dict):
    #     self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
    #         name, pretrained=pretrained)
    #     self.model.load_from_state_dict(state_dict)

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

import copy
import torch
import open_clip
from tqdm import tqdm
from utils.templates import get_templates
import sys
sys.path.append('..')
from models.CLIP_DLG import ClassificationHead, ImageEncoder

d = {'RN50':'openai',
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'laion2b_s32b_b82k',
    'convnext_base_w': 'laion2b_s13b_b82k'}

class Server(torch.nn.Module):
    def __init__(self, args, zeroshot=False):
        super().__init__()
        name = args.image_encoder_name
        pretrained = d[name]

        self.pretrained_model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        self.device = args.device
        self.pretrained_model.to(self.device)
        self.image_encoder = ImageEncoder(args, zeroshot).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def generate_cls_head(self, dataObject, data_name):
        print(f"build data {data_name} classification head")
        template = get_templates(data_name)

        logit_scale = self.pretrained_model.logit_scale
        self.pretrained_model.eval()
        self.pretrained_model.to(self.device)

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(dataObject.classnames):
                texts = []
                for t in template:
                    texts.append(t(classname))
                texts = open_clip.tokenize(texts).to(self.device) # tokenize
                embeddings = self.pretrained_model.encode_text(texts) # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)

                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()

                zeroshot_weights.append(embeddings)

            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()

            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

        return classification_head

    def generate_global_cls_head(self, cls_heads):
        global_cls_head = copy.deepcopy(cls_heads[0])
        for param in global_cls_head.parameters():
            param.data.zero_()
        for cls_head in cls_heads:
            for global_param, param in zip(global_cls_head.parameters(), cls_head.parameters()):
                global_param.data += param.data.clone()
        # mean
        for global_param in global_cls_head.parameters():
            global_param.data /= len(cls_heads)
        self.global_cls_head = global_cls_head
        self.global_cls_head.to(self.device)

    def freeze_except_global_adapter(self):
        for name, params in self.image_encoder.named_parameters():
            if 'global_adapter' not in name:
                params.requires_grad = False
        for params in self.global_cls_head.parameters():
            params.requires_grad = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel
import kornia
from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
from .xf import LayerNorm, Transformer
import math
import sys

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", use_patch_tokens=False):
        super().__init__()
        self.use_patch_tokens = use_patch_tokens
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                1,
            )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        if not self.use_patch_tokens:
            z = outputs.pooler_output
            z = z.unsqueeze(1)
            z = self.mapper(z)
        else:
            z = outputs.last_hidden_state
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)


class FrozenDinoV2Encoder(AbstractEncoder):
    """
     DINOv2 
     use_patch_tokens  CLS  CLS+Patch 
    """
    def __init__(self, 
                 weight_path="pretrained_models/dinov2_vitg14_pretrain.pth",
                 embed_dim=1536,   
                 out_dim=1024,     
                 use_patch_tokens=False,
                 device="cuda", 
                 freeze=True):
        super().__init__()
        
        sys.path.insert(0, "./dinov2")
        import hubconf
        dinov2 = hubconf.dinov2_vitg14(pretrained=False)
        
        if weight_path:
            print(f"[FrozenDinoV2Encoder] Loading weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            dinov2.load_state_dict(state_dict, strict=False)
        
        self.model = dinov2
        self.device = device
        self.embed_dim = embed_dim
        self.use_patch_tokens = use_patch_tokens
        
        if freeze:
            self.freeze()

        self.register_buffer('image_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.projector = nn.Linear(embed_dim, out_dim)
        self.final_ln = LayerNorm(out_dim)

    def freeze(self):
        self.model.eval()
        self.model.train = lambda mode=True: self.model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, list):
            image = torch.cat(image, 0)
        image = image.to(self.device)
        
        if image.shape[-1] != 224 or image.shape[-2] != 224:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        image = (image - self.image_mean) / self.image_std
        
        with torch.no_grad():
            features = self.model.forward_features(image)
        
        if not self.use_patch_tokens:
            z = features["x_norm_clstoken"].unsqueeze(1)
        else:
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            patch_tokens = features["x_norm_patchtokens"]
            z = torch.cat([cls_token, patch_tokens], dim=1)
        
        z = self.projector(z)
        z = self.final_ln(z)
        return z

    def encode(self, image):
        return self(image)

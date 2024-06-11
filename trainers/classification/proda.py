import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
from clip import clip
from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.classification.base_learner import VLBaseLearner
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ProDA',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        
        device = self.positional_embedding.device
        prompts = prompts.to(device)
        tokenized_prompts = tokenized_prompts.to(device)
        # self.positional_embedding = self.positional_embedding.to(device)
        # self.transformer = self.transformer.to(device)
        # self.ln_final = self.ln_final.to(device)

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PRODA.N_CTX
        # ctx_init = cfg.TRAINER.PRODA.CTX_INIT
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        n_prompt = cfg.TRAINER.PRODA.N_PROMPT
        prompt_bsz = cfg.TRAINER.PRODA.PROMPT_BS

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

 
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_prompt, n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        assert n_prompt % prompt_bsz == 0
        self.n_iter = int(n_prompt/prompt_bsz)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of prompts : {n_prompt}")
        print(f"Number of context words (tokens): {n_ctx}")

        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        classnames = [name.replace('_', ' ') for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if n_prompt >1:
            self.pos = [0 for _ in range(n_prompt//4)] + [1 for _ in range(n_prompt//4)] + [2 for _ in range(n_prompt//2)]
        else:
            self.pos = [2 for _ in range(n_prompt)]
        self.pos = torch.tensor(self.pos, device='cuda')

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            device = next(clip_model.parameters()).device
            print(device, 'device111') 
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim] 
        self.register_buffer('token_suffix', embedding[:, 1+n_ctx:, :]) # CLS, EOS, [n_cls, -1, ctx_dim] 

        nc_prompts = [prompt_prefix + '.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim] 
        self.register_buffer('nc_token_suffix', embedding[:, 1+n_ctx:, :]) # EOS, [n_cls, -1, ctx_dim] 



        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.ctx_dim = ctx_dim
        self.prompt_bsz = prompt_bsz
        self.iter_idx = 0



    def forward(self, infer=False):

        if self.n_iter > 1 and (not infer):
            if self.iter_idx == 0:
                self.select_idx = torch.randperm(self.n_prompt, device='cuda')
            batch_idx = self.select_idx[self.iter_idx*self.prompt_bsz: (self.iter_idx+1)*self.prompt_bsz]
            ctx = self.ctx[batch_idx]
            pos = self.pos[batch_idx]

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx = self.ctx
            pos = self.pos

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size, -1)

        n_cls = self.n_cls

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(n_cls, 1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)

        ctx_middle = ctx[pos==1]
        n_middle = ctx_middle.shape[0]
        prompts_middle = []
        half_n_ctx = self.n_ctx // 2
        for i in range(n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:, :].unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_middle, 1, dim)
                ctx_i_half1, # (1, n_middle, n_ctx//2, dim)
                class_i, # (1, n_middle, name_len, dim)
                ctx_i_half2, # (1, n_middle, n_ctx//2, dim)
                suffix_i # (1, n_middle, *, dim)
            ], dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        ctx_front = ctx[pos==0]
        n_front = ctx_front.shape[0]
        prompts_front = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_front, 1, dim)
                class_i, # (1, n_front, name_len, dim)
                ctx_i, # (1, n_front, n_ctx, dim)
                suffix_i # (1, n_front, *, dim)
            ], dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)

        prompts = torch.cat([prompts_end,prompts_middle, prompts_front], dim=1).view(prompt_size*n_cls, -1, self.ctx_dim)
        
        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_class = len(classnames)
        self.n_prompt = cfg.TRAINER.PRODA.N_PROMPT
        self.alpha = cfg.TRAINER.PRODA.ALPHA
        clip_model = clip_model.cuda()
        # text enoder
        self.text_encoder = TextEncoder(clip_model)

        # if torch.cuda.device_count() > 1:
        #     self.text_encoder = nn.DataParallel(self.text_encoder)

        # prompt learner
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)

        # image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype 







    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features.detach()

        n_class = self.n_class
        alpha = self.alpha
        
        if self.prompt_learner.training:
            assert label is not None
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts  = self.prompt_learner()
            n_prompt = text_prompt.shape[0]//n_class

            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, n_prompt, -1)
            text_mean = text_features.mean(dim=1)

            logit_scale = self.logit_scale.exp()
            # logit_scale = 160
            logits = logit_scale * image_features @ text_mean.t()

            batch_size = label.shape[0]
            
            text_features = text_features - text_mean.unsqueeze(1)
            diag_cov_martix = text_features.permute(2,0,1) @ text_features.permute(2,1,0)
            diag_cov_martix /= n_prompt + 1
            refined_logits = torch.einsum("bd, dik -> bik", [image_features**2, diag_cov_martix])

            sigma = refined_logits[torch.arange(batch_size), label, label].unsqueeze(-1) + \
                refined_logits[:, torch.arange(n_class), torch.arange(n_class) ] - \
                2 * refined_logits[torch.arange(batch_size), label, : ]

            logits += 0.5*(logit_scale**2)*sigma.view(-1, n_class)

            loss_upper = F.cross_entropy(logits, label)

            loss_m = None
            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.n_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            loss = loss_upper + alpha * loss_m

            return loss
        
        
        text_features = self.text_features
        logit_scale = self.logit_scale.exp()
        # logit_scale = 160
        logits = logit_scale * image_features @ text_features.t()


        return logits, image_features, text_features
    
    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner(infer=True)
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size = 1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, self.n_prompt, -1)
        text_features = text_features.mean(dim=1)
        self.text_features = text_features


@TRAINER_REGISTRY.register()
class ProDA(VLBaseLearner):
    """Prompt Distribution learning
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PRODA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.PRODA.PREC == "fp32" or cfg.TRAINER.PRODA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PRODA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            # self.model = nn.DataParallel(self.model)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PRODA.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

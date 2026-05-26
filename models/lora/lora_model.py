import torch
import torch.nn as nn
import clip

class PlainMultiheadAttentionLoRA(nn.Module):
    """
    Implements Low-Rank Adaptation for Multihead Attention layers.
    Injects learnable low-rank matrices into Q and V projections.
    """
    def __init__(self, module, embed_dim, r=4, lora_alpha=16, lora_dropout=0.0, param=['q', 'v']):
        super().__init__()
        self.module = module
        self.r = r
        self.scaling = lora_alpha / r
        self.param = param
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Low-rank matrices for Query projection
        if 'q' in self.param:
            self.lora_q_A = nn.Linear(embed_dim, r, bias=False)
            self.lora_q_B = nn.Linear(r, embed_dim, bias=False)
            nn.init.zeros_(self.lora_q_B.weight)
            nn.init.kaiming_uniform_(self.lora_q_A.weight, a=5**0.5)

        # Low-rank matrices for Value projection
        if 'v' in self.param:
            self.lora_v_A = nn.Linear(embed_dim, r, bias=False)
            self.lora_v_B = nn.Linear(r, embed_dim, bias=False)
            nn.init.zeros_(self.lora_v_B.weight)
            nn.init.kaiming_uniform_(self.lora_v_A.weight, a=5**0.5)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        # Execute the original frozen attention operation
        out = self.module(query, key, value, need_weights=need_weights, attn_mask=attn_mask)
        if isinstance(out, tuple):
            main_out = out[0]
            weights = out[1]
        else:
            main_out = out
            weights = None

        # Add the low-rank residual update (Delta W * x)
        if 'q' in self.param:
            q_lora = self.lora_q_B(self.lora_dropout(self.lora_q_A(query)))
            main_out = main_out + q_lora * self.scaling

        if 'v' in self.param:
            v_lora = self.lora_v_B(self.lora_dropout(self.lora_v_A(value)))
            main_out = main_out + v_lora * self.scaling

        return (main_out, weights) if weights is not None else main_out

def select_layers(position, total_layers):
    if position == 'all':
        return list(range(total_layers))
    elif position == 'top3':
        return list(range(max(0, total_layers - 3), total_layers))
    elif position == 'half-top':
        return list(range(total_layers // 2, total_layers))
    elif position == 'half-bottom':
        return list(range(total_layers // 2))
    elif position == 'bottom3':
        return list(range(min(3, total_layers)))
    else:
        raise ValueError(f"Unsupported LoRA position: {position}")

def mark_only_lora_as_trainable(model):
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def apply_lora(args, model):
    list_lora_layers = []

    # Text encoder (always 512 for ViT-B)
    if args.encoder in ['text', 'both']:
        text_layers = model.transformer.resblocks
        indices = select_layers(args.position, len(text_layers))
        for i in indices:
            old_attn = text_layers[i].attn
            new_attn = PlainMultiheadAttentionLoRA(
                old_attn, embed_dim=512, r=args.r, 
                lora_alpha=args.alpha, lora_dropout=args.dropout_rate, 
                param=args.params
            )
            text_layers[i].attn = new_attn
            list_lora_layers.append(new_attn)

    # Visual encoder (always 768 for ViT-B)
    if args.encoder in ['vision', 'both']:
        vision_layers = model.visual.transformer.resblocks
        indices = select_layers(args.position, len(vision_layers))
        for i in indices:
            old_attn = vision_layers[i].attn
            new_attn = PlainMultiheadAttentionLoRA(
                old_attn, embed_dim=768, r=args.r, 
                lora_alpha=args.alpha, lora_dropout=args.dropout_rate, 
                param=args.params
            )
            vision_layers[i].attn = new_attn
            list_lora_layers.append(new_attn)

    return list_lora_layers

def cls_acc(output, target, topk=1):
    with torch.no_grad():
        pred = output.topk(topk, 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        return 100 * acc / target.shape[0]

def clip_classifier(classnames, template, clip_model, device):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights

# Configuration and initialization for LoRA
class LoRAArgs:
    def __init__(self, encoder='vision', r=4, alpha=4, position='all'):
        self.encoder = encoder
        self.r = r
        self.alpha = alpha
        self.dropout_rate = 0.1
        self.params = ['q', 'v']
        self.position = position
        self.backbone = 'ViT-B/32'
        self.n_iters = 500  
        self.shots = 1      
        self.lr = 2e-4
        self.weight_decay = 1e-2
        self.betas = (0.9, 0.999)  

def get_lora_model(model, args, device):
    """
    Initializes the LoRA model by applying low-rank adaptations to the specified layers of the CLIP model.
    """
    apply_lora(args, model)
    mark_only_lora_as_trainable(model)
    model.to(device)
    return model

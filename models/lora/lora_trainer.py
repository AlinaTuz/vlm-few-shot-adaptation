import torch
import torch.nn.functional as F
from .lora_model import cls_acc, clip_classifier 

class LoRATrainer:
    """Trainer class for LoRA fine-tuning on CLIP. It handles the training loop, validation, and optimization of LoRA parameters."""
    def __init__(self, model, args, device, class_names):
        self.model = model
        self.args = args
        self.device = device
        self.class_names = class_names
        self.template = ["a photo of a {}."]
        
        # Precompute text features for all classes using the CLIP text encoder
        self.textual_features = clip_classifier(class_names, self.template, model, device)
        
        # Optimizer for LoRA parameters only
        lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            lora_params, 
            lr=args.lr, 
            weight_decay=args.weight_decay, 
            betas=args.betas
        )
        
        self.total_iters = args.n_iters
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.total_iters, eta_min=1e-6
        )
        
        self.device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        self.scaler = torch.amp.GradScaler('cuda') if self.device_type == "cuda" else None

    def train_epoch(self, train_loader, current_iter):
        """Trains the model for one epoch and returns training accuracy, loss, and updated iteration count."""
        self.model.train()
        acc_accum, loss_accum, samples = 0, 0, 0
        
        for images, target in train_loader:
            if current_iter >= self.total_iters:
                break
                
            images, target = images.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda"), dtype=torch.float16):
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ self.textual_features
                loss = F.cross_entropy(logits, target)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()

            acc_accum += cls_acc(logits.float(), target) * target.shape[0]
            loss_accum += loss.item() * target.shape[0]
            samples += target.shape[0]
            current_iter += 1
            
        return acc_accum / (samples if samples > 0 else 1), loss_accum / (samples if samples > 0 else 1), current_iter

    @torch.no_grad()
    def validate(self, val_loader):
        """Evaluates the model on the validation set and returns accuracy and loss."""
        self.model.eval()
        val_acc, val_loss, samples = 0, 0, 0
        
        for images, target in val_loader:
            images, target = images.to(self.device), target.to(self.device)
            with torch.amp.autocast(device_type=self.device_type, enabled=(self.device_type == "cuda"), dtype=torch.float16):
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ self.textual_features
                loss = F.cross_entropy(logits, target)

            val_acc += cls_acc(logits.float(), target) * target.shape[0]
            val_loss += loss.item() * target.shape[0]
            samples += target.shape[0]
            
        return val_acc / (samples if samples > 0 else 1), val_loss / (samples if samples > 0 else 1)

def train_lora(model, train_loader, val_loader, device, categories, args, 
               progress_bar=None, status_text=None, loss_chart=None, acc_chart=None):
    
    trainer = LoRATrainer(model, args, device, categories)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    current_iter = 0
    while current_iter < trainer.total_iters:
        train_acc, train_loss, current_iter = trainer.train_epoch(train_loader, current_iter)
        val_acc, val_loss = trainer.validate(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if progress_bar:
            progress_bar.progress(current_iter / trainer.total_iters)
        if status_text:
            status_text.text(f"LoRA Training: Iteration {current_iter}/{trainer.total_iters}")
        
        if loss_chart and acc_chart:
            loss_chart.add_rows({"Train Loss": [train_loss], "Val Loss": [val_loss]})
            acc_chart.add_rows({"Train Acc": [train_acc], "Val Acc": [val_acc]})
            
    return model, history

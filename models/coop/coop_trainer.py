import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .coop_model import CoopPromptLearner, TextEncoder
from data_utils import CLASS_NAMES

def train_coop(model, train_dataset, val_dataset, categories, device, 
               num_epochs=50, batch_size=128, lr=2e-3, t_max=100,
               progress_bar=None, status_text=None, loss_chart=None, acc_chart=None):
    
    """Trains the CoopPromptLearner on the given dataset while keeping CLIP frozen."""
    prompt_learner = CoopPromptLearner(categories, model).to(device)
    text_encoder = TextEncoder(model).to(device)
    
    # Freeze CLIP parameters
    for p in model.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    # Training only the prompt learner
    optimizer = torch.optim.SGD(prompt_learner.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    history = {"loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        prompt_learner.train()
        total_loss, total_correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompt_embeddings = prompt_learner() 
            token_ids = prompt_learner.tokenized_full_prompts
            
            text_features = text_encoder(prompt_embeddings, token_ids)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = model.logit_scale.exp() * (image_features @ text_features.T)
            
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_loss += loss.item() * images.size(0)

        scheduler.step()
        
        # Metrics
        avg_loss = total_loss / len(train_dataset)
        train_acc = total_correct / len(train_dataset)
        
        # Validation
        val_acc, _, _ = evaluate_coop(model, prompt_learner, text_encoder, val_dataset, categories, device, batch_size)

        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc * 100)
        history["val_acc"].append(val_acc * 100)

        if progress_bar: progress_bar.progress(epoch / num_epochs)
        if status_text: status_text.text(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        if loss_chart: loss_chart.line_chart(history["loss"])
        if acc_chart: acc_chart.line_chart({"Train": history["train_acc"], "Val": history["val_acc"]})

    return prompt_learner, history

@torch.no_grad()
def evaluate_coop(model, prompt_learner, text_encoder, dataset, categories, device, batch_size=128):
    """Evaluates the CoopPromptLearner on the given dataset."""
    model.eval()
    prompt_learner.eval()
    text_encoder.eval()
    
    y_true, y_pred = [], []
    category_to_index = {name: idx for idx, name in enumerate(categories)}
    prompt_embeddings = prompt_learner()
    token_ids = prompt_learner.tokenized_full_prompts
    text_features = text_encoder(prompt_embeddings, token_ids)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    correct_count = 0
    
    for images, labels in data_loader:
        images = images.to(device)        
        target_labels = torch.tensor(
            [category_to_index[CLASS_NAMES[l.item()]] for l in labels], 
            dtype=torch.long, 
            device=device
        )

        # Image features
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate similarities and predictions
        similarities = image_features @ text_features.T
        predictions = similarities.argmax(dim=-1)

        # Metrics
        correct_count += (predictions == target_labels).sum().item()
        y_true.extend(target_labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

    accuracy = correct_count / len(dataset)
    return accuracy, y_true, y_pred

import torch
import clip
from tqdm import tqdm
from data_utils import CLASS_NAMES 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

@torch.no_grad()
def eval_zero_shot(model, dataset, categories, batch_size, device, label=""):
    """Model evaluation in zero-shot setting."""
    model.eval()
    y_true, y_pred = [], []
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    # Prompt tokenizer
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)

    # Text embeddings
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()
        image, target = image.to(device), target.to(device)

        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.T
        predicted_class = logits.argmax(dim=-1)
        
        correct_predictions += (predicted_class == target).sum().item()
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted_class.cpu().numpy())

    accuracy = correct_predictions / len(dataset)
    return accuracy, y_true, y_pred

def harmonic_mean(base_acc, novel_acc):
    if (base_acc + novel_acc) == 0: return 0
    return 2 * (base_acc * novel_acc) / (base_acc + novel_acc)

@torch.no_grad()
def predict_single_zero_shot(model, image_tensor, device):
    """Predict the class of a single image using the zero-shot CLIP model."""
    model.eval()
    
    # Use class names as prompts
    text_inputs = clip.tokenize(
        [f"a photo of a {name}, a type of flower." for name in CLASS_NAMES]
    ).to(device)
    
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    image_features = model.encode_image(image_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity and get top-1 prediction
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    
    return CLASS_NAMES[indices[0]], values[0].item()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm, 
        annot=False, 
        fmt='d', 
        cmap='Blues', 
        ax=ax,
        xticklabels=class_names, 
        yticklabels=class_names
    )
    
    ax.set_xlabel('Predicted labels', fontsize=12)
    ax.set_ylabel('True labels', fontsize=12)
    ax.set_title(title, fontsize=15)
    
    plt.tight_layout()
    return fig
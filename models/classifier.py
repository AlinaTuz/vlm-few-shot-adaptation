import torch
import clip

@torch.no_grad()
def predict_single_coop(model, prompt_learner, text_encoder, image_tensor, class_names):
    """
    Function to predict the class of a single image using the CoOp model.
    This function processes the image and generates adaptive prompts based on the image features.
    """
    model.eval()
    prompt_learner.eval()
    text_encoder.eval()
    
    # Extract image features using the CLIP image encoder
    image_features = model.encode_image(image_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Generate text features using the trained PromptLearner
    prompts = prompt_learner() 
    tokenized_prompts = prompt_learner.tokenized_full_prompts
    
    # Encode the prompts using the CLIP text encoder
    text_features = text_encoder(prompts, tokenized_prompts)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity between image features and text features
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T
    
    # Obtain probabilities through Softmax
    probs = logits.softmax(dim=-1)
    
    # Find the top-1 result
    confidence, index = probs[0].topk(1)
    
    return class_names[index[0].item()], confidence[0].item()

@torch.no_grad()
def predict_single_cocoop(model, prompt_learner, text_encoder, image_tensor, class_names):
    """
    Function to predict the class of a single image using the CoCoOp model.
    This function processes the image and generates adaptive prompts based on the image features.
    """
    model.eval()
    prompt_learner.eval()
    text_encoder.eval()
    
    # Extract image features using the CLIP image encoder
    image_features = model.encode_image(image_tensor) # (1, visual_dim)
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Generate adaptive prompts based on image features
    all_prompts = prompt_learner(image_features) 
    
    # Get the token IDs (they are fixed for each class)
    tokenized_prompts = prompt_learner.tokenized_full_prompts 
    
    # Processing through TextEncoder
    b_size, n_classes, seq_len, e_dim = all_prompts.shape
    flat_prompts = all_prompts.view(b_size * n_classes, seq_len, e_dim)
    text_features = text_encoder(flat_prompts, tokenized_prompts) 
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity between image features and text features
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features_norm @ text_features.T 
    
    # Obtain probabilities through Softmax
    probs = logits.softmax(dim=-1)
    confidence, index = probs[0].topk(1)
    
    return class_names[index[0].item()], confidence[0].item()

@torch.no_grad()
def predict_single_lora(model, image_tensor, device,class_names):
    """
    Prediction for a single image using the LoRA model. 
    LoRA modifies the attention layers, but the overall prediction process remains similar to zero-shot.
    """
    model.eval()
    
    # Tokenize prompts for all classes
    text_inputs = clip.tokenize([f"a photo of a {name}, a type of flower." for name in class_names]).to(device)
    
    # Image Features
    image_features = model.encode_image(image_tensor)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity and get top-1 prediction   
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(1)
    
    return class_names[indices[0]], values[0].item()



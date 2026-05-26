import torch
import torch.nn as nn
import clip
from collections import OrderedDict

class TextEncoder(nn.Module):
    """
    Encodes a sequence of prompt and class‑name token embeddings into a single
    text feature vector that we will use for similarity computation with image features.
    """
    def __init__(self, model):
        super().__init__()

        # Projection from hidden dimension to embedding dimension
        self.text_projection = model.text_projection

        # Positional embeddings with shape (sequence_length, embedding_dim) (L, D)
        self.positional_embedding = model.positional_embedding

        # CLIP’s pretrained transformer for text
        self.text_transformer = model.transformer

        # Final layer normalization
        self.layer_norm = model.ln_final

        self.dtype = model.dtype

    def forward(self, token_embeddings, token_ids):
        """
        token_embeddings: tensor of shape (batch_size, sequence_length, embedding_dim) (B, L, D)
                          contains both prompt tokens and class-name token embeddings
        token_ids:        tensor of shape (batch_size, sequence_length) (B, L)
                          contains integer token indices; used to locate the end‑of‑sequence (EOS) token

        returns:
        text_features:    tensor of shape (batch_size, embedding_dim) (B, D)
                          pooled representation of each prompt sequence
        """

        # Add positional encodings so the transformer can distinguish token order
        hidden_states = token_embeddings + self.positional_embedding.type(self.dtype)

        # The CLIP transformer expects input in (L, B, D)
        hidden_states = hidden_states.permute(1, 0, 2)

        # Run the sequence through the frozen transformer to get context‑aware features
        hidden_states = self.text_transformer(hidden_states)

        # Convert back to (B, L, D) for pooling
        hidden_states = hidden_states.permute(1, 0, 2)

        hidden_states = self.layer_norm(hidden_states).type(self.dtype)

        # Locate the end‑of‑sequence (EOS) token for each example
        eos_positions = token_ids.argmax(dim = -1) # (B)

        # Gather the hidden state at the EOS position for each batch element
        batch_range = torch.arange(hidden_states.size(0))
        pooled_states = hidden_states[batch_range, eos_positions, :]

        # Project the pooled state into the same embedding space as CLIP’s text encoder
        text_features = pooled_states @ self.text_projection # (B, D)
        return text_features
    
class CoCoopPromptLearner(torch.nn.Module):
    """Implements image-conditional prompts by combining shared context vectors 
    with a MetaNet that generates per-image adjustments."""
    def __init__(self, class_names, model, number_of_context_tokens = 16, initial_prompt = None):
        super().__init__()

        # number of classes and the size of each embedding vector
        device = next(model.parameters()).device
        self.total_classes = len(class_names)
        self.visual_dimension = model.visual.output_dim
        self.embedding_dimension = model.ln_final.weight.shape[0]
        self.dtype = model.dtype

        # initializing the learnable context tokens from the initial prompt or intializing them randomly
        if initial_prompt:
            number_of_context_tokens = len(initial_prompt.split())
            prompt_token_ids = clip.tokenize(initial_prompt).to(device)
            with torch.no_grad():
                prompt_embeddings = model.token_embedding(prompt_token_ids).type(self.dtype)
            self.initial_prompt = initial_prompt
            # we skip the SOS token and take the next tokens as our context basis
            context_token_embeddings = prompt_embeddings[0, 1 : 1 + number_of_context_tokens, :]
        else:
            self.initial_prompt = " ".join(["X"] * number_of_context_tokens)
            context_token_embeddings = torch.randn(number_of_context_tokens, self.embedding_dimension, dtype = self.dtype, device = device) * 0.02

        # these are the context token parameters we will update along with the ones of the MLP, since CLIP's parameters will be freezed
        self.context_vector_tokens = torch.nn.Parameter(context_token_embeddings)

       # Constructing MetaNet: a lightweight MLP to generate conditioning vectors
        # Features are reduced by 16x before expanding back to embedding dimension
        reduced_dim = self.visual_dimension // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("first_linear_layer", nn.Linear(self.visual_dimension, reduced_dim)),
            ("relu_activation",    nn.ReLU(inplace = True)),
            ("second_linear_layer",nn.Linear(reduced_dim, self.embedding_dimension)),
        ]))

        self.meta_net = self.meta_net.to(self.dtype)

        # Full prompts: [SOS] + context tokens + "class name + [EOS]"
        class_names = [name.replace("_", " ") for name in class_names]
        full_prompts = [f"{self.initial_prompt} {name}." for name in class_names]
        tokenized_full_prompts = torch.cat([clip.tokenize(p) for p in full_prompts]).to(device)

        with torch.no_grad():
            full_embeddings = model.token_embedding(tokenized_full_prompts).type(self.dtype)
        prefix_embeddings = full_embeddings[:, : 1, :] # [SOS]
        suffix_embeddings = full_embeddings[:, 1 + number_of_context_tokens:, :] # class + [EOS]

        # Fixed as buffers
        self.register_buffer("prefix_embeddings", prefix_embeddings)
        self.register_buffer("suffix_embeddings", suffix_embeddings)
        self.register_buffer("tokenized_class_prompts", tokenized_full_prompts)

        self.tokenized_full_prompts = tokenized_full_prompts

    def forward(self, image_features):
        """Generates a batch of conditional prompts adjusted for each input image."""
        # 1. Generate a conditioning vector from image features
        conditioning_vector = self.meta_net(image_features) # (batch_size, embedding_dim)
        conditioning_vector = conditioning_vector.unsqueeze(1) # (batch_size, 1, embedding_dim)

        # 2. Add the image-specific conditioning to shared context tokens
        base_context = self.context_vector_tokens.unsqueeze(0) # (1, num_ctx, embedding_dim)
        adapted_context = base_context + conditioning_vector # (batch_size, num_ctx, embedding_dim)

        # Retrieve the fixed SOS prefix and class-name suffix embeddings
        sos_prefix  = self.prefix_embeddings # (num_classes, 1, embedding_dim)
        eos_suffix  = self.suffix_embeddings # (num_classes, suffix_len, embedding_dim)

        # 3. Build full prompts for all classes per image
        batch_prompts = []
        for image_ctx in adapted_context: # iterate (batch_size) times
            repeated_ctx = image_ctx.unsqueeze(0).expand(self.total_classes, -1, -1) # (num_classes, num_ctx, embedding_dim)
            per_image_prompts = torch.cat([sos_prefix, repeated_ctx, eos_suffix], dim = 1) # (num_classes, seq_length, embedding_dim)
            batch_prompts.append(per_image_prompts)

        # Single tensor of shape (batch_size, num_classes, seq_length, embedding_dim)
        all_prompts = torch.stack(batch_prompts, dim = 0)
        return all_prompts
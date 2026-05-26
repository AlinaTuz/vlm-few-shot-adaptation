import torch
import torch.nn as nn
import clip

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

class CoopPromptLearner(torch.nn.Module):
    """Learns a small set of context token embeddings per class while keeping the rest of CLIP frozen."""
    def __init__(self, class_names, model, number_of_context_tokens = 16, initial_prompt = None):
        super().__init__()

        device = next(model.parameters()).device
        self.total_classes = len(class_names)
        self.embedding_dimension = model.ln_final.weight.shape[0]
        self.dtype = model.dtype

        # Initialization of context tokens
        if initial_prompt:
            # Initialization from a specific string (e.g., "a photo of a")
            number_of_context_tokens = len(initial_prompt.split())
            prompt_token_ids = clip.tokenize(initial_prompt).to(device)
            with torch.no_grad():
                prompt_embeddings = model.token_embedding(prompt_token_ids).type(self.dtype)
            self.initial_prompt = initial_prompt
            context_token_embeddings = prompt_embeddings[0, 1 : 1 + number_of_context_tokens, :]
        else:
            # Random initialization
            self.initial_prompt = " ".join(["X"] * number_of_context_tokens)
            context_token_embeddings = torch.randn(number_of_context_tokens, self.embedding_dimension, dtype = self.dtype, device = device) * 0.02

        self.context_vector_tokens = torch.nn.Parameter(context_token_embeddings)

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

    def forward(self):
        """Concatenates [SOS], learnable context_tokens, and [class name + EOS]."""
        context = self.context_vector_tokens
        context = context.unsqueeze(0).expand(self.total_classes, -1, -1)
        prompt_embeddings = torch.cat([self.prefix_embeddings, context, self.suffix_embeddings], dim = 1)
        return prompt_embeddings
    

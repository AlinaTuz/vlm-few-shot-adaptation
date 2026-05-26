import clip
import torch
import streamlit as st

@st.cache_resource
def load_clip_model():
    """Initializes the CLIP model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    return model, preprocess, device
import streamlit as st
from PIL import Image
import torch
import wikipedia
import clip
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
wikipedia.set_lang("en")
from data_utils import CLASS_NAMES, get_data, base_novel_categories, split_data
from models.base_clip import load_clip_model
from models.zero_shot import predict_single_zero_shot
from models.coop.coop_model import CoopPromptLearner, TextEncoder
from models.coop.coop_trainer import train_coop
from models.cocoop.cocoop_model import CoCoopPromptLearner
from models.lora.lora_model import get_lora_model, LoRAArgs
from models.lora.lora_trainer import train_lora
from models.classifier import predict_single_coop, predict_single_cocoop, predict_single_lora
import json
from sklearn.metrics import confusion_matrix

# config
st.set_page_config(layout="wide", page_title="VLM Adaptation")

# load CLIP model and preprocess
model, preprocess, device = load_clip_model()

@st.cache_resource
def prepare_datasets(_preprocess):
    """Orchestrates the data preparation pipeline: loading, splitting, and labeling."""
    # Fetch raw data 
    train_raw, val_raw, test_raw = get_data(transform=_preprocess)
    
    # Identify base and novel categories
    base_idx, novel_idx = base_novel_categories(train_raw)
    base_classes = [CLASS_NAMES[i] for i in base_idx]
    novel_classes = [CLASS_NAMES[i] for i in novel_idx]
    
    # Create subsets for training and testing
    train_base, _ = split_data(train_raw, base_idx)
    val_base, _ = split_data(val_raw, base_idx)
    test_base, test_novel = split_data(test_raw, base_idx)
    
    return {
        "train_base": train_base,
        "val_base": val_base,
        "test_base": test_base,
        "test_novel": test_novel,
        "base_classes": base_classes,
        "novel_classes": novel_classes
    }

@st.cache_resource
def load_coop_resources(_model, _device):
    """Initializes CoOp components and loads pre-trained weights."""
    prompt_learner = CoopPromptLearner(
        CLASS_NAMES, 
        _model, 
        number_of_context_tokens=32  
    )
    text_encoder = TextEncoder(_model)
    
    try:
        state_dict = torch.load("models/coop/coop_weights.pth", map_location=_device)
        prompt_learner.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("File coop_weights.pth not found!")
        
    prompt_learner.to(_device)
    text_encoder.to(_device)
    
    return prompt_learner, text_encoder

@st.cache_resource
def load_cocoop_resources(_model, _device):
    """Initializes CoCoOp components and loads pre-trained weights."""
    prompt_learner = CoCoopPromptLearner(
        CLASS_NAMES, 
        _model, 
        number_of_context_tokens=16
    )
    text_encoder = TextEncoder(_model)
    
    try:
        state_dict = torch.load("models/cocoop/cocoop_weights.pth", map_location=_device)
        prompt_learner.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error("File cocoop_weights.pth not found!")
        
    prompt_learner.to(_device)
    text_encoder.to(_device)
    prompt_learner.eval()
    text_encoder.eval()
    
    return prompt_learner, text_encoder

@st.cache_resource
def load_lora_resources(_device):
    """Initializes the LoRA-adapted CLIP model and loads pre-trained weights."""
    model_lora, _ = clip.load("ViT-B/16", device=_device)
    args = LoRAArgs(encoder='vision', r=4, alpha=4, position='all')
    
    model_lora = get_lora_model(model_lora, args, _device)
    
    try:
        checkpoint = torch.load("models/lora/lora_weights.pth", map_location=_device)
        model_lora.load_state_dict(checkpoint)
    except FileNotFoundError:
        st.sidebar.error("File lora_weights.pth not found")
        
    model_lora.eval()
    return model_lora

# Loading resources for all models at the start 
coop_learner, coop_encoder = load_coop_resources(model, device)
cocoop_learner, cocoop_encoder = load_cocoop_resources(model, device)
lora_model = load_lora_resources(device)

def load_history(path_str):
    """Loads training history from a JSON file."""
    path = Path(path_str)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def get_cm(option, folder, device, loader):
    cm_path = Path("models") / folder / f"cm_{folder}.npy"
    if cm_path.exists():
        return np.load(cm_path)

    if option == "Zero-shot":
        return np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))

    st.info(f"🔄 Calculating confusion matrix for {option}...")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            
            if option == "CoOp":
                logits = model.encode_image(images) @ coop_learner().t()
            elif option == "CoCoOp":
                img_features = model.encode_image(images)
                logits = torch.stack([f @ cocoop_learner(f).t() for f in img_features])
            elif option == "LoRA":
                logits = lora_model(images)
            else:
                continue

            # Collect predictions and labels for confusion matrix
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_preds:
        return np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cm_path, cm)
    
    return cm

def load_confusion_matrix(file_path: Path, num_classes):
    """Loads the confusion matrix from a .npy file or creates a zero matrix."""
    if file_path.exists():
        return np.load(file_path)
    return np.zeros((num_classes, num_classes))

def display_metrics_table(selected_option):
    """
    Loads results and displays a table.
    Paths are specified relative to the project's root folder.
    """
    methods = {
        "Zero-shot": "zero_shot",
        "CoOp": "coop",
        "CoCoOp": "cocoop",
        "LoRA": "lora"
    }
    
    all_rows = []
    
    for display_name, folder in methods.items():
        res_path = Path("models") / folder / f"{folder}_results.json"
        
        if res_path.exists():
            with open(res_path, "r") as f:
                data = json.load(f)
                all_rows.append({
                    "Method": display_name,
                    "Base Accuracy (%)": round(data.get("base", 0), 2),
                    "Novel Accuracy (%)": round(data.get("novel", 0), 2),
                    "Harmonic Mean (%)": round(data.get("harmonic", 0), 2)
                })
    
    if not all_rows:
        st.warning("Results files not found in the models/ directory.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Filtering based on user selection
    if selected_option == "All Methods":
        display_df = df
    else:
        display_df = df[df["Method"] == selected_option]

    if not display_df.empty:
        st.subheader(f"📈 Results: {selected_option}")
        st.table(display_df)
    
    return display_df
   
with st.sidebar:
    ### Sidebar for navigation and configuration.
    st.title("⚙️Menu")
    
    # Main navigation
    page = st.radio("Choose a section:", ["Image Analysis", "Metrics and Graphs", "Training"])
    
    st.divider()
    
    # Model configuration options will depend on the selected page
    st.subheader("Model Configuration")
    
    if page == "Image Analysis":
        option = st.selectbox("Choose a method:", ["Zero-shot", "CoOp", "CoCoOp", "LoRA", "All Methods"])
        st.info("🔍 In this section you can perform analysis of flower images.")
        
    elif page == "Metrics and Graphs":
        option = st.selectbox("Choose a method for testing:", ["Zero-shot", "CoOp", "CoCoOp", "LoRA", "All Methods"])
        st.info("📊 In this section you can analyze the performance of adapted models.")
        
    else:
        option = st.selectbox("Choose a method for training:", ["CoOp", "LoRA"])
        st.info("📚 In this section you can perform full training of the model.")

    st.divider()
    st.caption("This app compares the effectiveness of CLIP adaptation on limited data.")

if page == "Image Analysis":
    ### Main page for image analysis. Users can upload an image and 
    ### select which method(s) to apply for classification.
    st.title("Image Analysis with CLIP")

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.header("🖼️ Input Data")
        uploaded_file = st.file_uploader("Choose a flower image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Uploaded Image")

    def draw_result_card(method_name, flower_name, confidence, status="info"):
        with st.container(border=True): 
            if status == "success":
                st.success(f"**{method_name}**")
            else:
                st.info(f"**{method_name}**")
                
            st.write(f"🌸 **{flower_name.title()}**") 
            st.write(f"Confidence: {confidence:.2f}%")
            st.progress(confidence / 100)

    # Results
    with col_output:
        st.header("🔍 Analysis Results")
        
        if uploaded_file:
            img_tensor = preprocess(image).unsqueeze(0).to(device)
            
            results = {}

            if option in ["Zero-shot", "All Methods"]:
                with st.spinner('CLIP analyzing...'):
                    label, conf = predict_single_zero_shot(model, img_tensor, device)
                    results["Zero-shot"] = (label, conf, "info")

            if option in ["CoOp", "All Methods"]:
                with st.spinner('CoOp analyzing...'):
                    label, conf = predict_single_coop(model, coop_learner, coop_encoder, img_tensor, CLASS_NAMES)
                    results["CoOp (Optimized)"] = (label, conf, "success")

            if option in ["CoCoOp", "All Methods"]:
                with st.spinner('CoCoOp analyzing...'):
                    label, conf = predict_single_cocoop(model, cocoop_learner, cocoop_encoder, img_tensor, CLASS_NAMES)
                    results["CoCoOp (Optimized)"] = (label, conf, "success")

            if option in ["LoRA", "All Methods"]:
                with st.spinner('LoRA analyzing...'):
                    label, conf = predict_single_lora(lora_model, img_tensor, device, CLASS_NAMES)
                    results["LoRA (Fine-tuned)"] = (label, conf, "success")

            if option == "All Methods":
                for name, data in results.items():
                    draw_result_card(name, data[0], round(data[1] * 100, 1), data[2])
                st.caption(f"⏱️ Real-time Mode (Inference)")

                # Select the best result for Wikipedia reference
                best_model = max(results, key=lambda x: results[x][1])
                st.session_state.last_label = results[best_model][0]
            else:
                name = list(results.keys())[0]
                data = results[name]
                draw_result_card(name, data[0], round(data[1] * 100, 1), data[2])
                st.session_state.last_label = data[0]
                st.caption(f"⏱️ Real-time Mode (Inference)")

    st.divider() 
    tab1, tab2 = st.tabs(["📖 Wikipedia reference", "✔️ Method Details"])

    with tab1:
        st.subheader("Botanical Reference (Wikipedia)")
        current_flower = st.session_state.get('last_label', None)
        
        if uploaded_file and current_flower:
            with st.spinner(f"Fetching data for '{current_flower}'..."):
                try:
                    import wikipedia
                    wikipedia.set_user_agent("FlowersClassificationApp/1.0 (contact@example.com)")
                    wikipedia.set_lang("en")                   
                    
                    try:
                        page = wikipedia.page(current_flower, auto_suggest=False)
                    except wikipedia.exceptions.DisambiguationError as de:
                        st.warning(f"Multiple meanings found for '{current_flower}'. Trying the first botanical alternative...")
                        page = wikipedia.page(de.options[0], auto_suggest=False)

                    st.markdown(f"# {page.title}")                    
                    
                    col_text, col_img = st.columns([2, 1])
                    with col_text:
                        st.markdown(f"**Overview:** {page.summary}")

                    if len(page.content) > len(page.summary) + 150:
                        st.markdown("### Detailed Description")
                        
                        full_content = page.content
                        if "== See also ==" in full_content:
                            full_content = full_content.split("== See also ==")[0]
                        elif "== References ==" in full_content:
                            full_content = full_content.split("== References ==")[0]
                        
                        formatted_content = full_content.replace("==", "###")                     
                        st.write(formatted_content[:5000]) 
                    else:
                        st.caption("_No additional detailed sections available for this species on Wikipedia._")
                    
                    st.divider()
                    st.link_button("Read Full Article on Wikipedia ↗", page.url, use_container_width=True)
                        
                except wikipedia.exceptions.PageError:
                    st.warning(f"Could not find a specific Wikipedia page for '{current_flower}'.")
                except Exception as e:
                    st.error(f"Notice: Information formatting issue. {e}")
        else:
            st.info("Upload an image and run analysis to see botanical details.")
    with tab2:
        st.write(f"Architecture Description {option}:")
        if option == "Zero-shot":
            st.write("The original CLIP ViT-B/16 is used without further fine-tuning.")
        

elif page == "Metrics and Graphs":
    ### Page for displaying performance metrics and visualizations for the different methods.
    st.header("Experimental Results")
    df_results = display_metrics_table(option)

    if not df_results.empty:
        st.divider()        
        st.subheader(f"📊 Comparative Chart: {option}")
        
        fig_bar = go.Figure()
        metrics = [("Base Accuracy (%)", "#1f77b4"), 
                   ("Novel Accuracy (%)", "#ff7f0e"), 
                   ("Harmonic Mean (%)", "#2ca02c")]
        
        for col_name, color in metrics:
            fig_bar.add_trace(go.Bar(
                x=df_results["Method"], 
                y=df_results[col_name],
                name=col_name,
                marker_color=color,
                text=df_results[col_name],
                textposition='auto'
            ))

        fig_bar.update_layout(barmode='group', height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

        folders_to_show = []
        if option == "All Methods":
            folders_to_show = ["coop", "cocoop", "lora"]
        elif option in ["CoOp", "CoCoOp", "LoRA"]:
            folders_to_show = [option.lower()]

        for folder in folders_to_show:
            hist = load_history(f"models/{folder}/{folder}_history.json")
            if hist:
                st.subheader(f"🔄 Dynamic of Training: {folder.upper()}")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(pd.DataFrame({"Train Acc": hist['train_acc'], "Val Acc": hist['val_acc']}))
                with col2:
                    st.line_chart(pd.DataFrame({"Train Loss": hist['train_loss'], "Val Loss": hist['val_loss']}))

        # Confusion Matrix
        st.divider()
        st.subheader(f"🧩 Confusion Matrix ({option})")

        folder_map = {"Zero-shot": "zero_shot", "CoOp": "coop", "CoCoOp": "cocoop", "LoRA": "lora", "All Methods": "lora"}
        current_folder = folder_map.get(option, "zero_shot") 
        cm_path = Path("models") / current_folder / f"cm_{current_folder}.npy"

        cm_data = load_confusion_matrix(cm_path, len(CLASS_NAMES))

        if np.sum(cm_data) > 0:
            num_classes_in_matrix = cm_data.shape[0]

            if len(CLASS_NAMES) >= num_classes_in_matrix:
                display_names = CLASS_NAMES[:num_classes_in_matrix]
            else:
                display_names = CLASS_NAMES + [f"Class {i}" for i in range(len(CLASS_NAMES), num_classes_in_matrix)]

            cm_norm = cm_data.astype('float') / cm_data.sum(axis=1, keepdims=True)
            
            fig_cm = px.imshow(
                cm_norm, 
                x=display_names, 
                y=display_names,
                color_continuous_scale='Viridis',
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Score")
            )
            
            fig_cm.update_layout(
                xaxis = dict(tickmode = 'linear', dtick = 2),
                yaxis = dict(tickmode = 'linear', dtick = 2),
                width=900, height=900
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)

        else:
            st.warning(f"Data for confusion matrix {option} is missing or the file is empty.")


else:
    ### Page for training the CoOp and LoRA models. Users can configure training parameters and monitor progress.
    st.header(f"Training Adaptive Model {option}")
    if device.lower() == "cpu":
            st.warning("⚠️ **Warning:** You are using CPU for training. "
                       "This will be significantly slower than GPU (CUDA). "
                       "Consider using fewer epochs or a machine with a GPU.")
    
    col_info, col_action = st.columns([1, 1.5])
    
    # Upload and prepare datasets
    with st.spinner("Loading Flowers102 dataset..."):
        data_dict = prepare_datasets(preprocess)
    
    train_base = data_dict["train_base"]
    val_base = data_dict["val_base"]
    base_classes = data_dict["base_classes"]
    
    with col_info:
        st.subheader("📝 Configuration")
        
        if option == "CoOp":
            st.markdown(f"""
            **Method:** Context Optimization (CoOp)  
            **Tokens (N_CTX):** `16`  
            **Learning Rate:** `2e-3`  
            **Optimizer:** `SGD`  
            **Base:** `CLIP ViT-B/16`
            """)
            epochs = st.number_input("Number of epochs:", min_value=1, max_value=200, value=10)
            
        elif option == "LoRA":
            st.markdown(f"""
            **Method:** Low-Rank Adaptation (LoRA)  
            **Rank (Rank):** `4`  
            **Alpha:** `4`  
            **Learning Rate:** `2e-4`  
            **Optimizer:** `AdamW`  
            **Base:** `CLIP ViT-B/32`
            """)
            epochs = st.number_input("Number of epochs for training:", min_value=1, max_value=50, value=5)

        save_dir_str = st.text_input("Folder for saving weights:", value=f"models/{option.lower()}")

    with col_action:
        st.subheader("Control")
        st.write(f"Device for computation: **{device.upper()}**")
        
        btn_start = st.button(f"Start Training {option}", use_container_width=True)
        if st.button("🛑 Stop/Reset", use_container_width=True):
            st.rerun()

    st.divider()

    if btn_start:
        st.session_state.training_done = False
        st.subheader(f"📈 Monitoring {option}")
        status_msg = st.empty()
        progress_bar = st.progress(0)
        
        col_loss, col_acc = st.columns(2)
        loss_chart = col_loss.empty()
        acc_chart = col_acc.empty()

        try:
            if option == "CoOp":
                trained_model, history = train_coop(
                    model=model, 
                    train_dataset=train_base, 
                    val_dataset=val_base, 
                    categories=base_classes,
                    device=device,
                    num_epochs=epochs,
                    progress_bar=progress_bar,
                    status_text=status_msg,
                    loss_chart=loss_chart,
                    acc_chart=acc_chart
                )
                save_filename = "coop_weights.pth"

            elif option == "LoRA":                
                lora_args = LoRAArgs(r=4, alpha=4)
                lora_args.n_iters = epochs 
                model = get_lora_model(model, lora_args, device)
                import pandas as pd
                l_chart = loss_chart.line_chart(pd.DataFrame(columns=["Train Loss", "Val Loss"]))
                a_chart = acc_chart.line_chart(pd.DataFrame(columns=["Train Acc", "Val Acc"]))

                trained_model, history = train_lora(
                    model=model,
                    train_loader=torch.utils.data.DataLoader(train_base, batch_size=32, shuffle=True),
                    val_loader=torch.utils.data.DataLoader(val_base, batch_size=32),
                    device=device,
                    categories=base_classes,
                    args=lora_args,
                    progress_bar=progress_bar, 
                    status_text=status_msg,
                    loss_chart=l_chart,
                    acc_chart=a_chart
                )
                save_filename = "lora_weights.pth"

            # Save the trained model
            save_folder = Path(save_dir_str)
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / save_filename            
            torch.save(trained_model.state_dict(), save_path)            
            st.session_state.training_done = True
            st.session_state.last_save_path = str(save_path)
            st.success(f"Training {option} completed!")

        except Exception as e:
            st.error(f"❌ Error during training {option}: {e}")

    # Look results
    if st.session_state.get("training_done"):
        st.divider()
        st.subheader("💾 Export Results")
        p = Path(st.session_state.last_save_path)
        if p.exists():
            with open(p, "rb") as file:
                st.download_button(
                    label=f"Download Weights {option} (.pth)",
                    data=file,
                    file_name=f"{option.lower()}_flowers102.pth",
                    mime="application/octet-stream",
                    use_container_width=True
                )
# -*- coding: utf-8 -*-
"""Agri_interface_Krishika.py

Modernized UI rebranding to 'Krishika' with premium CSS and dynamic model loading.
Fixes for Gradio 6.0+ compatibility.
"""

import os
import joblib
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')
IMG_SIZE = 224

# --- 2. GLOBAL STYLING ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

body { 
    background: linear-gradient(135deg, #f0f7f4 0%, #ffffff 100%) !important; 
    font-family: 'Outfit', sans-serif !important; 
}

.gradio-container { 
    max-width: 850px !important; 
}

.title-section { 
    text-align: center; 
    padding: 30px 10px; 
}

.input-box {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    border: 1px solid #e0eadd !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
}

.output-box {
    background: white !important;
    border-radius: 20px !important;
    padding: 2px !important;
    border: none !important;
}

button.primary {
    background: linear-gradient(45deg, #2e7d32, #43a047) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    height: 50px !important;
}

button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(46, 125, 50, 0.4) !important;
}
"""

# --- 3. MODEL SETUP ---
print("Initializing Krishika Engine (VGG19)...")
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# --- 4. DYNAMIC MODEL LOADING ---
loaded_models = {}
class_mappings = {
    'tomato': ['bacterial_spot', 'early_blight', 'Fusarium Wilt', 'healthy_leaf', 'late_blight', 'leaf_curl', 'leaf_miner', 'leaf_mold', 'septoria_leaf', 'spider mites', 'verticillium wilt'],
    'apple': ['Apple black_spot', 'Apple Brown_spot', 'Apple Normal'],
    'apricot': ['Apricot blight leaf disease', 'Apricot Normal', 'Apricot shot_hole'],
    'beans': ['Bean bean rust image', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean shot_hole'],
    'brinjal': ['Healthy Leaf', 'Insect Pest Disease', 'Leaf Spot Disease', 'Mosaic Virus Disease', 'White Mold Disease', 'Wilt Disease'],
    'cashew': ['Cashew healthy', 'Cashew leaf miner', 'Cashew red rust'],
    'cherry': ['Cherry brown_spot', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry purple leaf spot', 'Cherry_shot hole disease'],
    'corn': ['Corn Fungal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Corn Normal leaf'],
    'cucumber': ['Anthracnose', 'Bacterial Wilt', 'Belly Rot', 'Downy Mildew', 'Fresh Cucumber', 'Fresh Leaf', 'Gummy Stem Blight', 'Pythium Fruit Rot'],
    'fig': ['Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf'],
    'grape': ['Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease'],
    'guava': ['Canker', 'Curling', 'Healthy', 'Leaf Spot', 'Nutritional Deficiency', 'Powdery Mildew', 'Rust'],
    'lychee': ['Algal Spot Indirect', 'Anthracnose Cloudy', 'Dry Leaves', 'Entomosporium Spot', 'Leaf Mites Direct', 'Mayetiola PostRain'],
    'potato': ['Fungi', 'Healthy', 'Nematode'],
    'radish': ['Black leaf spot', 'Downey mildew', 'flea beetle', 'Fresh leaf', 'Mosaic virus'],
    'rice': ['bacterial_leaf_blight', 'brown_spot', 'leaf_blast', 'rice_healthy'],
    'walnut': ['Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut leaf gall mite', 'Walnut Normal_leaf', 'Walnut Shot_hole']
}

if os.path.exists(OUTPUT_ROOT):
    for plant_folder in os.listdir(OUTPUT_ROOT):
        plant_path = os.path.join(OUTPUT_ROOT, plant_folder)
        if os.path.isdir(plant_path):
            model_file = f"{plant_folder}_model.pkl"
            full_model_path = os.path.join(plant_path, model_file)
            if os.path.exists(full_model_path):
                try:
                    loaded_models[plant_folder] = joblib.load(full_model_path)
                    print(f"✅ Active: {plant_folder}")
                except Exception as e:
                    print(f"⚠️ Load Error ({plant_folder}): {e}")

# --- 5. REMEDY DATABASE ---
agri_database = {
    'tomato': {
         'late_blight': {'cause': 'Phytophthora infestans', 'remedy': 'Apply Chlorothalonil or Mancozeb. Use Trichoderma for organic control.', 'impact': 'Severe fruit decay in 48h.', 'severity': 'high'},
         'leaf_curl': {'cause': 'Virus (ToLCV) spread by whiteflies.', 'remedy': 'Use Neem oil spray and sticky traps. Prune infected parts.', 'impact': 'Complete growth stunting.', 'severity': 'high'},
         'healthy_leaf': {'cause': 'None', 'remedy': 'Continue regular irrigation and NPK schedule.', 'impact': 'Optimal crop health.', 'severity': 'healthy'}
    },
    'apple': {
         'apple black_spot': {'cause': 'Fungal infestation.', 'remedy': 'Increase air circulation and use Copper based fungicides.', 'impact': 'Market value drops significantly.', 'severity': 'high'},
         'apple normal': {'cause': 'None', 'remedy': 'Continue seasonal pruning.', 'impact': 'Premium fruit yield.', 'severity': 'healthy'}
    }
}

# --- 6. MODERN REPORT GENERATOR ---
def generate_krishika_report(disease_name, crop_type):
    data = agri_database.get(crop_type.lower(), {}).get(disease_name.lower(), {})
    cause = data.get('cause', 'Biological analysis in progress.')
    remedy = data.get('remedy', 'Consult local krishi vigyan kendra for detailed recommendation.')
    impact = data.get('impact', 'Yield impact currently being calculated.')
    severity = data.get('severity', 'medium')

    colors = {
        'healthy': {'solid': '#2e7d32', 'light': '#e8f5e9', 'icon': '🌱', 'status': 'STABLE'},
        'high': {'solid': '#d32f2f', 'light': '#ffebee', 'icon': '⚠️', 'status': 'DANGER'},
        'medium': {'solid': '#f57c00', 'light': '#fff3e0', 'icon': '🔍', 'status': 'MONITOR'}
    }
    
    config = colors.get(severity, colors['medium'])

    return f"""
    <div style='background: {config['light']}; border: 1px solid {config['solid']}44; border-radius: 20px; padding: 25px; font-family: "Outfit", sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.03);'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
            <div style='display: flex; align-items: center; gap: 12px;'>
                <span style='font-size: 32px;'>{config['icon']}</span>
                <h2 style='color: {config['solid']}; margin: 0; font-size: 24px; font-weight: 700;'>{disease_name.replace("_", " ").title()}</h2>
            </div>
            <span style='background: {config['solid']}; color: white; padding: 5px 15px; border-radius: 30px; font-size: 12px; font-weight: 700;'>{config['status']}</span>
        </div>
        
        <div style='display: grid; gap: 18px;'>
            <div style='background: white; padding: 15px; border-radius: 12px;'>
                <span style='display: block; color: #888; font-size: 11px; font-weight: 600; text-transform: uppercase;'>Biological Cause</span>
                <span style='color: #2c3e50; font-size: 15px;'>{cause}</span>
            </div>
            <div style='background: white; padding: 15px; border-radius: 12px;'>
                <span style='display: block; color: #888; font-size: 11px; font-weight: 600; text-transform: uppercase;'>Recommended Action</span>
                <span style='color: #2c3e50; font-size: 15px;'>{remedy}</span>
            </div>
            <div style='background: white; padding: 15px; border-radius: 12px; border: 1px dashed {config['solid']}88;'>
                <span style='display: block; color: #888; font-size: 11px; font-weight: 600; text-transform: uppercase;'>Economic Impact</span>
                <span style='color: #2c3e50; font-size: 15px; font-weight: 600;'>{impact}</span>
            </div>
        </div>
    </div>
    """

# --- 7. LOGIC ---
def predict_disease(crop_type, image):
    if image is None: return "<div style='text-align:center; padding: 20px;'>Please upload a leaf image to begin.</div>"
    if crop_type not in loaded_models: return "<div style='text-align:center; padding: 20px;'>Selected model currently unavailable.</div>"

    try:
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_batch = np.expand_dims(img_resized, axis=0)
        features = feature_extractor.predict(preprocess_input(img_batch.astype(float)), verbose=0)
        
        model = loaded_models[crop_type]
        pred_idx = model.predict(features)[0]
        
        plant_lower = crop_type.lower()
        if plant_lower in class_mappings and pred_idx < len(class_mappings[plant_lower]):
            disease_name = class_mappings[plant_lower][pred_idx]
        else:
            disease_name = f"Unknown Class ({pred_idx})"

        return generate_krishika_report(disease_name, crop_type)
    except Exception as e:
        return f"<div style='color:red; padding: 20px;'>System Error: {str(e)}</div>"

# --- 8. UI RUNTIME ---
if __name__ == "__main__":
    with gr.Blocks(title="Krishika AI") as app:
        # Title Header (Using Markdown with HTML for styling)
        gr.Markdown(f"""
        <div style='text-align: center; padding: 30px 10px;'>
            <h1 style='font-family: "Outfit", sans-serif; font-weight: 700; color: #1b5e20; font-size: 3rem; margin-bottom: 5px; letter-spacing: -1px;'>Krishika AI</h1>
            <p style='color: #43a047; font-weight: 400; font-size: 1.1rem; margin-top: 0;'>Smart Agriculture Diagnosis • Powered by VGG19 & SVM</p>
        </div>
        """)
        
        # Main Layout
        with gr.Row():
            with gr.Column(scale=4, elem_classes="input-box"):
                crop_in = gr.Dropdown(
                    choices=list(loaded_models.keys()), 
                    label="📂 Target Crop", 
                    value=list(loaded_models.keys())[0] if loaded_models else None
                )
                img_in = gr.Image(label="📸 Capture/Upload Leaf Image", type="numpy")
                btn = gr.Button("⚡ Start Deep Diagnosis", variant="primary", elem_classes="primary")
            
            with gr.Column(scale=5, elem_classes="output-box"):
                report_out = gr.HTML(
                    label="Diagnosis Result", 
                    value="<div style='text-align:center; padding-top: 100px; color: #999; font-style: italic;'>Final diagnosis report will appear here.</div>"
                )

        btn.click(predict_disease, inputs=[crop_in, img_in], outputs=report_out)

    print("Krishika AI is running...")
    app.launch(share=True, theme=gr.themes.Soft(primary_hue="green"), css=CSS)
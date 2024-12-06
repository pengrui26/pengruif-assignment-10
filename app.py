import os
import io
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import open_clip

app = Flask(__name__)

# Load model and tokenizer
model_name = "ViT-B-32-quickgelu"
pretrained = "openai"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.eval()

tokenizer = open_clip.get_tokenizer(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load image embeddings
df = pd.read_pickle("image_embeddings.pickle")
embeddings = np.stack(df['embedding'])  # [N, D]
file_names = df['file_name'].tolist()

def get_text_embedding(text_query):
    text_tokens = tokenizer([text_query])
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, p=2, dim=1)
    return text_emb.cpu().numpy()  # [1, D]

def get_image_embedding_from_pil(pil_image):
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_emb = model.encode_image(image_input)
        image_emb = F.normalize(image_emb, p=2, dim=1)
    return image_emb.cpu().numpy()  # [1, D]

def apply_pca(emb_matrix, k):
    pca = PCA(n_components=k)
    emb_reduced = pca.fit_transform(emb_matrix)
    return emb_reduced, pca

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None

    if request.method == 'POST':
        text_query = request.form.get('text_query', '').strip()
        lam = request.form.get('lam', '0.5').strip()
        pca_k = request.form.get('pca_k', '').strip()

        if lam == '':
            lam = 0.5
        else:
            lam = float(lam)

        image_file = request.files.get('image_query', None)
        have_image = (image_file is not None and image_file.filename != '')
        have_text = (text_query != '')

        # Determine if PCA is applied
        use_pca = (pca_k != '')
        if use_pca:
            pca_k = int(pca_k)
            emb_reduced, pca_model = apply_pca(embeddings, pca_k)
        else:
            emb_reduced = embeddings

        query_vec = None

        if have_text and not have_image:
            # Text-only
            text_emb = get_text_embedding(text_query)
            if use_pca:
                text_emb = pca_model.transform(text_emb)
            query_vec = text_emb

        elif have_image and not have_text:
            # Image-only
            image_bytes = image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_emb = get_image_embedding_from_pil(pil_image)
            if use_pca:
                img_emb = pca_model.transform(img_emb)
            query_vec = img_emb

        elif have_image and have_text:
            # Combined
            image_bytes = image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_emb = get_image_embedding_from_pil(pil_image)
            text_emb = get_text_embedding(text_query)

            if use_pca:
                img_emb = pca_model.transform(img_emb)
                text_emb = pca_model.transform(text_emb)

            query_vec = lam * text_emb + (1.0 - lam) * img_emb

        # If query_vec could not be formed, just show the form again (no results)
        if query_vec is not None:
            similarities = cosine_similarity(query_vec, emb_reduced)  # [1, N]
            sim_scores = similarities[0]
            top5_indices = np.argsort(sim_scores)[::-1][:5]
            results = []
            for idx in top5_indices:
                results.append({
                    "file_name": file_names[idx],
                    "similarity": float(sim_scores[idx])
                })

    # Render index.html with or without results
    return render_template('index.html', results=results)

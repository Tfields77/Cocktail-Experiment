import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
import os
import pyheif

# Set the path for the Tesseract executable
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load Data
def load_data():
    return pd.read_csv('space_cocktail.csv')

cocktails = load_data()

if cocktails is not None:
    st.write("Data loaded successfully")
else:
    st.error("Failed to load data")

# Clean Text
def clean_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return text

cocktails = cocktails.fillna('')
space_cocktail = cocktails.applymap(lambda x: clean_text(x))

# Train Word2Vec model
@st.cache_resource
def train_word2vec(data):
    sentences = data.apply(lambda row: ' '.join([str(row[col]) for col in data.columns if 'ingredient' in col]).split(), axis=1)
    model = Word2Vec(sentences, vector_size=200, window=10, min_count=1, sg=1)
    return model

model = train_word2vec(space_cocktail)

# Compute TF-IDF weights
def compute_tfidf(data):
    ingredients = data.apply(lambda row: ' '.join([str(row[col]) for col in data.columns if 'ingredient' in col]), axis=1)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(ingredients)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = dict(zip(feature_names, tfidf_matrix.sum(axis=0).A1))
    return vectorizer, tfidf_dict

vectorizer, tfidf_dict = compute_tfidf(space_cocktail)

# Vectorize recipes with TF-IDF and Word2Vec
def get_recipe_vector(ingredients, model, tfidf_dict, vector_size):
    recipe_vector = np.zeros((1, vector_size))
    count = 0
    for ingredient in ingredients:
        if ingredient in model.wv.key_to_index:
            weight = tfidf_dict.get(ingredient, 1)
            recipe_vector += model.wv[ingredient] * weight
            count += weight
    if count != 0:
        recipe_vector /= count
    # Normalize the vector
    norm = np.linalg.norm(recipe_vector)
    if norm != 0:
        recipe_vector = recipe_vector / norm
    return recipe_vector

def recommend_drinks(liked_ingredients, model, tfidf_dict, data, top_n=3):
    liked_vector = get_recipe_vector(liked_ingredients, model, tfidf_dict, model.vector_size)
    data['recipe_vector'] = data.apply(lambda row: get_recipe_vector(
        [row[col] for col in ['ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6'] if row[col] != ''], 
        model, tfidf_dict, model.vector_size), axis=1)

    similarities = data['recipe_vector'].apply(lambda vec: cosine_similarity(liked_vector, vec.reshape(1, -1))[0][0])
    data['similarity'] = similarities

    top_recommendations = data.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recommendations[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6', 'instructions', 'similarity']]

# Extract text from image
def extract_text_from_image(image):
    if image.format == "HEIC":
        heif_file = pyheif.read(image)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
    text = pytesseract.image_to_string(image)
    return text

# Filter dataset based on liked cocktails
def filter_dataset(cocktails, liked_cocktails):
    filtered_cocktails = cocktails[cocktails['name'].isin(liked_cocktails)]
    return filtered_cocktails

# Streamlit app layout
st.title("The Cocktail-Experiment")

# Input cocktails you like
st.header("Enter Cocktails You Like")
liked_cocktails_input = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita, moscow mule, old-fashioned, manhattan, negroni")

# Upload a picture of a menu
st.header("Upload a Picture of a Menu")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff", "heic"])

submit = st.button(label='Submit')

if submit:
    liked_cocktails = [cocktail.strip() for cocktail in liked_cocktails_input.split(",")]
    temporary_dataset = filter_dataset(space_cocktail, liked_cocktails)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Menu', use_column_width=True)
        menu_text = extract_text_from_image(image)
        st.write("Extracted text from the menu:")

        # Split menu items by newline and remove empty lines
        menu_items = [item.strip() for item in menu_text.split("\n") if item]
        
        # Parse menu items and their ingredients
        parsed_menu_items = []
        current_item = ""
        for item in menu_items:
            if item.isupper() and len(item.split()) <= 4:
                if current_item:
                    parsed_menu_items.append(current_item)
                current_item = item
            else:
                current_item += " " + item
        if current_item:
            parsed_menu_items.append(current_item)

        # Extract cocktail names and ingredients
        menu_ingredients = []
        for item in parsed_menu_items:
            parts = item.split(":")
            if len(parts) > 1:
                item_name = parts[0].strip()
                item_ingredients = parts[1].strip().split(", ")
                menu_ingredients.append((item_name, item_ingredients))

        st.write("### Top 3 recommendations based on your preferences:")
        recommendations = []
        for item_name, item_ingredients in menu_ingredients:
            item_recommendations = recommend_drinks(item_ingredients, model, tfidf_dict, temporary_dataset, top_n=3)
            for index, row in item_recommendations.iterrows():
                recommendations.append({
                    'menu_item': item_name,
                    'recommendation': row['name'],
                    'ingredients': ', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']])),
                    'instructions': row['instructions'],
                    'similarity': row['similarity']
                })

        sorted_recommendations = sorted(recommendations, key=lambda x: x['similarity'], reverse=True)[:3]
        
        for rec in sorted_recommendations:
            st.write(f"**Menu Item: {rec['menu_item']}**")
            st.write(f"**Recommended Drink: {rec['recommendation']}**")
            st.write(f"Ingredients: {rec['ingredients']}")
            st.write(f"Instructions: {rec['instructions']}")
            st.write(f"Similarity: {rec['similarity']:.2f}")
            st.write("---")





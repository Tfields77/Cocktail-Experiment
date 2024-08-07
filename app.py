import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
import os

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

def recommend_drinks(liked_ingredients, model, tfidf_dict, data, top_n=5):
    liked_vector = get_recipe_vector(liked_ingredients, model, tfidf_dict, model.vector_size)
    data['recipe_vector'] = data.apply(lambda row: get_recipe_vector(
        [row[col] for col in ['ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6'] if row[col] != ''], 
        model, tfidf_dict, model.vector_size), axis=1)

    similarities = data['recipe_vector'].apply(lambda vec: cosine_similarity(liked_vector, vec.reshape(1, -1))[0][0])
    data['similarity'] = similarities

    top_recommendations = data.sort_values(by='similarity', ascending=False).head(top_n)
    top_recommendations = top_recommendations.sample(frac=1).reset_index(drop=True)

    return top_recommendations[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6', 'instructions', 'similarity']]

# Extract text from image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Streamlit app layout
st.title("The Cocktail-Experiment")

# Input cocktails you like
st.header("Enter Cocktails You Like")
liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita")
submit_cocktails = st.button(label='Submit Cocktails')
if submit_cocktails:
    liked_ingredients = [ingredient.strip() for cocktail in liked_cocktails.split(",") for ingredient in cocktail.split()]
    recommendations = recommend_drinks(liked_ingredients, model, tfidf_dict, space_cocktail)
    st.write("### Recommendations based on cocktails you like:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['name']}**")
        st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
        st.write(f"Instructions: {row['instructions']}")
        st.write(f"Similarity: {row['similarity']:.2f}")
        st.write("---")

# Upload a picture of a menu
st.header("Upload a Picture of a Menu")
uploaded_image = st.file_uploader("Choose an image...", type="jpg")
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Menu', use_column_width=True)
    menu_text = extract_text_from_image(image)
    st.write("Extracted text from the menu:")
    st.write(menu_text)

    menu_items = [item.strip() for item in menu_text.split("\n") if item]
    st.write("Menu Items:")
    st.write(menu_items)

    if submit_cocktails and menu_items:
        st.write("### Recommendations for the menu:")
        for item in menu_items:
            item_ingredients = [ingredient.strip() for ingredient in item.split()]
            item_recommendations = recommend_drinks(item_ingredients, model, tfidf_dict, space_cocktail)
            st.write(f"### Recommendations for {item}:")
            for index, row in item_recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4', 'ingredient-5', 'ingredient-6']]))}")
                st.write(f"Instructions: {row['instructions']}")
                st.write(f"Similarity: {row['similarity']:.2f}")
                st.write("---")

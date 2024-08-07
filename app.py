import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract

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
    text = pytesseract.image_to_string(image)
    return text

# Filter dataset based on liked cocktails
def filter_dataset(data, liked_cocktails):
    liked_cocktails = [clean_text(cocktail) for cocktail in liked_cocktails]
    filtered_data = data[data['name'].apply(lambda x: clean_text(x) in liked_cocktails)]
    return filtered_data

# Streamlit app layout
st.title("The Cocktail-Experiment")

st.header("Enter Cocktails You Like")
liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita, moscow mule, old-fashioned, manhattan, negroni")
uploaded_image = st.file_uploader("Upload a Picture of a Menu", type=["jpg", "jpeg", "png", "bmp", "tiff", "heic", "tif"])
submit_button = st.button(label='Submit')

if submit_button:
    liked_ingredients = [ingredient.strip() for cocktail in liked_cocktails.split(",") for ingredient in cocktail.split()]
    temporary_dataset = filter_dataset(space_cocktail, liked_cocktails.split(","))
    st.write("Temporary dataset based on your likes:")
    st.write(temporary_dataset)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Menu', use_column_width=True)
        menu_text = extract_text_from_image(image)
        st.write("Extracted text from the menu:")
        st.write(menu_text)
        
        menu_items = [item.strip() for item in menu_text.split("\n") if item]
        st.write("Menu Items:")
        st.write(menu_items)

        recommendations = []
        for item in menu_items:
            item_ingredients = [ingredient.strip() for ingredient in item.split()]
            item_recommendations = recommend_drinks(item_ingredients, model, tfidf_dict, temporary_dataset)
            recommendations.extend(item_recommendations.to_dict('records'))

        if recommendations:
            st.write("Top 3 recommendations based on your preferences:")
            for rec in recommendations[:3]:
                st.write(f"Menu Item: {rec['name']}")
                st.write(f"Recommended Drink: {rec['name']}")
                st.write(f"Ingredients: {', '.join(filter(None, [rec['ingredient-1'], rec['ingredient-2'], rec['ingredient-3'], rec['ingredient-4'], rec['ingredient-5'], rec['ingredient-6']]))}")
                st.write(f"Instructions: {rec['instructions']}")
                st.write(f"Similarity: {rec['similarity']:.2f}")
                st.write("---")







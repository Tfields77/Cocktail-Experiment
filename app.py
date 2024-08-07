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
    top_recommendations = top_recommendations.sample(frac=1).reset_index(drop=True)
    
    return top_recommendations[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6', 'instructions', 'similarity']]

# Extract text from image
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Parse menu text to create DataFrame
def parse_menu_text(menu_text):
    menu_lines = menu_text.split("\n")
    parsed_data = {'name': [], 'ingredient-1': [], 'ingredient-2': [], 'ingredient-3': [], 'ingredient-4': [], 'ingredient-5': [], 'ingredient-6': []}
    
    for line in menu_lines:
        if line.strip() and len(line.split(',')) > 1:
            parts = line.split(',')
            parsed_data['name'].append(parts[0].strip())
            ingredients = parts[1:]
            for i, ingredient in enumerate(ingredients[:6]):
                parsed_data[f'ingredient-{i+1}'].append(ingredient.strip())
            for i in range(len(ingredients), 6):
                parsed_data[f'ingredient-{i+1}'].append(None)
        else:
            continue
    
    return pd.DataFrame(parsed_data)

# Streamlit app layout
st.title("The Cocktail-Experiment")

st.header("Enter Cocktails You Like")
liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita, moscow mule, old-fashioned, manhattan, negroni")
submit_cocktails = st.button(label='Submit')
uploaded_image = st.file_uploader("Upload a Picture of a Menu", type=["jpg", "jpeg", "png", "bmp", "tiff", "heic", "tif"])

if submit_cocktails:
    liked_ingredients = [ingredient.strip() for cocktail in liked_cocktails.split(",") for ingredient in cocktail.split()]
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Menu', use_column_width=True)
        menu_text = extract_text_from_image(image)
        st.write("Extracted text from the menu:")
        st.write(menu_text)
        
        parsed_menu_df = parse_menu_text(menu_text)
        st.write("Parsed Menu DataFrame:")
        st.write(parsed_menu_df)
        
        recommendations = recommend_drinks(liked_ingredients, model, tfidf_dict, parsed_menu_df)
        st.write("Top 3 recommendations based on your preferences:")
        for index, row in recommendations.iterrows():
            st.write(f"Menu Item: {row['name']}")
            st.write(f"Recommended Drink: {row['name']}")
            st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
            st.write(f"Instructions: {row['instructions']}")
            st.write(f"Similarity: {row['similarity']:.2f}")
            st.write("---")
    else:
        st.write("Please upload a menu image to get recommendations.")






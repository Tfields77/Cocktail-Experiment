import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import time
# Load data
@st.cache
def load_data():
    file_path = 'space_cocktail.csv'  # Adjust the path to your CSV file
    return pd.read_csv(file_path)

cocktails = load_data()

# Check if data is loaded
if cocktails is not None:
    st.write("Data loaded successfully")
else:
    st.error("Failed to load data")

# Preprocess data
def clean_text(text):
    if isinstance(text, str):
        return text.strip().lower()
    return text

# Fill NaN values with a placeholder, e.g., ''
cocktails = cocktails.fillna('')

space_cocktail = cocktails.applymap(lambda x: clean_text(x))

# Train word2vec model
def train_word2vec(data):
    sentences = data.apply(lambda row: ' '.join([str(row[col]) for col in data.columns if 'ingredient' in col]).split(), axis=1)
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model

model = train_word2vec(space_cocktail)

# Vectorize recipes
def get_recipe_vector(ingredients, model, vector_size):
    recipe_vector = np.zeros((1, vector_size))
    count = 0
    for ingredient in ingredients:
        if ingredient in model.wv.key_to_index:
            recipe_vector += model.wv[ingredient]
            count += 1
    if count != 0:
        recipe_vector /= count
    return recipe_vector

def recommend_drinks(liked_ingredients, model, data, top_n=5):
    liked_vector = get_recipe_vector(liked_ingredients, model, model.vector_size)
    data['recipe_vector'] = data.apply(lambda row: get_recipe_vector(
        [row[col] for col in ['ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6'] if row[col] != ''], 
        model, model.vector_size), axis=1)
    
    similarities = data['recipe_vector'].apply(lambda vec: cosine_similarity(liked_vector, vec.reshape(1, -1))[0][0])
    data['similarity'] = similarities
    
    recommendations = data.sort_values(by='similarity', ascending=False).head(top_n)
    return recommendations[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6', 'instructions', 'similarity']]

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Streamlit app layout with sequential input fields
st.title("The Cocktail-Experiment")

def go_to_next_step():
    st.session_state.step += 1

# Step 1: Input for cocktails you like
if st.session_state.step == 1:
    with st.form(key='cocktails_form'):
        liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita")
        submit_cocktails = st.form_submit_button(label='Submit Cocktails')
        if submit_cocktails:
            st.session_state.liked_cocktails = [cocktail.strip() for cocktail in liked_cocktails.split(",")]
            go_to_next_step()

# Step 2: Input for ingredients you like
if st.session_state.step == 2:
    with st.form(key='ingredients_form'):
        liked_ingredients = st.text_input("Enter ingredients you like (comma-separated):", "vodka, lime, mint")
        submit_ingredients = st.form_submit_button(label='Submit Ingredients')
        if submit_ingredients:
            st.session_state.liked_ingredients = [ingredient.strip() for ingredient in liked_ingredients.split(",")]
            go_to_next_step()

# Step 3: Input for flavors you like
if st.session_state.step == 3:
    with st.form(key='flavors_form'):
        liked_flavors = st.text_input("Enter flavors you like (comma-separated):", "sweet, sour, spicy")
        submit_flavors = st.form_submit_button(label='Submit Flavors')
        if submit_flavors:
            st.session_state.liked_flavors = [flavor.strip() for flavor in liked_flavors.split(",")]
            go_to_next_step()

# Step 4: Input for drinks you're curious about
if st.session_state.step == 4:
    with st.form(key='curious_drinks_form'):
        curious_drinks_input = st.text_area("Enter ingredients for drinks you're curious about, one line per drink. Put commas after each ingredient:")
        submit_curious_drinks = st.form_submit_button(label='Submit Curious Drinks')
        if submit_curious_drinks:
            st.session_state.curious_drinks = [line.split(",") for line in curious_drinks_input.split("\n") if line]
            go_to_next_step()

# Step 5: Display recommendations
if st.session_state.step == 5:
    recommendations = recommend_drinks(st.session_state.liked_ingredients, model, space_cocktail)
    st.write("### Recommendations based on ingredients you like:")
    for index, row in recommendations.iterrows():
        st.write(f"**{row['name']}**")
        st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
        st.write(f"Instructions: {row['instructions']}")
        st.write(f"Similarity: {row['similarity']:.2f}")
        st.write("---")
    
    flavor_recommendations = recommend_drinks(st.session_state.liked_flavors, model, space_cocktail)
    st.write("### Recommendations based on flavors you like:")
    for index, row in flavor_recommendations.iterrows():
        st.write(f"**{row['name']}**")
        st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
        st.write(f"Instructions: {row['instructions']}")
        st.write(f"Similarity: {row['similarity']:.2f}")
        st.write("---")
    
    if st.session_state.curious_drinks:
        st.write("### Recommendations for drinks you're curious about:")
        for drink_ingredients in st.session_state.curious_drinks:
            drink_recommendations = recommend_drinks(drink_ingredients, model, space_cocktail)
            for index, row in drink_recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
                st.write(f"Instructions: {row['instructions']}")
                st.write(f"Similarity: {row['similarity']:.2f}")
                st.write("---")

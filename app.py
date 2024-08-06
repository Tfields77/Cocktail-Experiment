import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# Load data
@st.cache
def load_data():
    file_path = 'space_cocktail.csv'  # Adjust the path to your CSV file
    return pd.read_csv(file_path)

cocktails = load_data()

if cocktails is not None:
    st.write("Data loaded successfully")
else:
    st.error("Failed to load data")

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
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
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
            weight = tfidf_dict.get(ingredient, 0)
            recipe_vector += model.wv[ingredient] * weight
            count += weight
    if count != 0:
        recipe_vector /= count
    return recipe_vector

def recommend_drinks(liked_ingredients, model, tfidf_dict, data, top_n=5):
    liked_vector = get_recipe_vector(liked_ingredients, model, tfidf_dict, model.vector_size)
    data['recipe_vector'] = data.apply(lambda row: get_recipe_vector(
        [row[col] for col in ['ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6'] if row[col] != ''], 
        model, tfidf_dict, model.vector_size), axis=1)
    
    similarities = data['recipe_vector'].apply(lambda vec: cosine_similarity(liked_vector, vec.reshape(1, -1))[0][0])
    data['similarity'] = similarities
    
    # Get top_n similar drinks and shuffle the results
    top_recommendations = data.sort_values(by='similarity', ascending=False).head(top_n)
    top_recommendations = top_recommendations.sample(frac=1).reset_index(drop=True)
    
    return top_recommendations[['name', 'ingredient-1', 'ingredient-2', 'ingredient-3', 'ingredient-4', 'ingredient-5', 'ingredient-6', 'instructions', 'similarity']]

# Streamlit app layout with tabs for input and recommendations
st.title("The Cocktail-Experiment")

tab1, tab2, tab3, tab4 = st.tabs(["Cocktails You Like", "Ingredients You Like", "Flavors You Like", "Drinks You're Curious About"])

with tab1:
    st.header("Cocktails You Like")
    with st.form(key='cocktails_form'):
        liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita")
        submit_cocktails = st.form_submit_button(label='Submit Cocktails')
        if submit_cocktails:
            st.session_state.liked_cocktails = [cocktail.strip() for cocktail in liked_cocktails.split(",")]
            recommendations = recommend_drinks(st.session_state.liked_cocktails, model, tfidf_dict, space_cocktail)
            st.write("### Recommendations based on cocktails you like:")
            for index, row in recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
                st.write(f"Instructions: {row['instructions']}")
                st.write(f"Similarity: {row['similarity']:.2f}")
                st.write("---")

with tab2:
    st.header("Ingredients You Like")
    with st.form(key='ingredients_form'):
        liked_ingredients = st.text_input("Enter ingredients you like (comma-separated):", "vodka, lime, mint")
        submit_ingredients = st.form_submit_button(label='Submit Ingredients')
        if submit_ingredients:
            st.session_state.liked_ingredients = [ingredient.strip() for ingredient in liked_ingredients.split(",")]
            recommendations = recommend_drinks(st.session_state.liked_ingredients, model, tfidf_dict, space_cocktail)
            st.write("### Recommendations based on ingredients you like:")
            for index, row in recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
                st.write(f"Instructions: {row['instructions']}")
                st.write(f"Similarity: {row['similarity']:.2f}")
                st.write("---")

with tab3:
    st.header("Flavors You Like")
    with st.form(key='flavors_form'):
        liked_flavors = st.text_input("Enter flavors you like (comma-separated):", "sweet, sour, spicy")
        submit_flavors = st.form_submit_button(label='Submit Flavors')
        if submit_flavors:
            st.session_state.liked_flavors = [flavor.strip() for flavor in liked_flavors.split(",")]
            recommendations = recommend_drinks(st.session_state.liked_flavors, model, tfidf_dict, space_cocktail)
            st.write("### Recommendations based on flavors you like:")
            for index, row in recommendations.iterrows():
                st.write(f"**{row['name']}**")
                st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
                st.write(f"Instructions: {row['instructions']}")
                st.write(f"Similarity: {row['similarity']:.2f}")
                st.write("---")

with tab4:
    st.header("Drinks You're Curious About")
    with st.form(key='curious_drinks_form'):
        liked_cocktails = st.text_input("Enter cocktails you like (comma-separated):", "mojito, margarita")
        curious_drinks_input = st.text_area("Enter ingredients for drinks you're curious about, one line per drink. Put commas after each ingredient:")
        submit_curious_drinks = st.form_submit_button(label='Submit Curious Drinks')
        if submit_curious_drinks:
            st.session_state.liked_cocktails = [cocktail.strip() for cocktail in liked_cocktails.split(",")]
            st.session_state.curious_drinks = [line.split(",") for line in curious_drinks_input.split("\n") if line]
            for drink_ingredients in st.session_state.curious_drinks:
                drink_recommendations = recommend_drinks(drink_ingredients, model, tfidf_dict, space_cocktail)
                st.write(f"### Recommendations for drink with ingredients: {', '.join(drink_ingredients)}")
                for index, row in drink_recommendations.iterrows():
                    st.write(f"**{row['name']}**")
                    st.write(f"Ingredients: {', '.join(filter(None, [row['ingredient-1'], row['ingredient-2'], row['ingredient-3'], row['ingredient-4'], row['ingredient-5'], row['ingredient-6']]))}")
                    st.write(f"Instructions: {row['instructions']}")
                    st.write(f"Similarity: {row['similarity']:.2f}")
                    st.write("---")

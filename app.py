import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the cocktail dataset
space_cocktail = pd.read_csv('space_cocktail.csv')

# Initialize a session state to keep track of recommended drinks
if 'recommended_drinks' not in st.session_state:
    st.session_state['recommended_drinks'] = []

def vectorize_ingredients(data):
    data['recipe_vector'] = data.apply(lambda row: ' '.join([str(ingredient) for ingredient in filter(None, [
        row['ingredient-1'], row['ingredient-2'], row['ingredient-3'],
        row['ingredient-4'], row['ingredient-5'], row['ingredient-6']
    ])]), axis=1)
    return data

def get_recommendations(liked_cocktails, dataset):
    # Remove already recommended drinks
    dataset = dataset[~dataset['name'].isin(st.session_state['recommended_drinks'])]
    
    if dataset.empty:
        st.warning("No new drinks to recommend based on the current session.")
        return pd.DataFrame()  # Return an empty DataFrame if all drinks have been recommended
    
    # Filter the dataset to only include the cocktails the user likes
    liked_cocktails_df = dataset[dataset['name'].str.lower().isin(liked_cocktails)]
    
    if liked_cocktails_df.empty:
        st.warning("None of the liked cocktails are in the dataset.")
        return pd.DataFrame()

    # Vectorize the entire dataset and the liked cocktails
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(dataset['recipe_vector'])
    
    liked_vector = vectorizer.transform(liked_cocktails_df['recipe_vector'])
    
    # Convert the liked_vector to a numpy array
    liked_vector = np.asarray(liked_vector.mean(axis=0)).flatten()
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectorized_data, [liked_vector])
    
    # Add similarity scores to the dataset
    dataset['similarity'] = cosine_sim.flatten()
    
    # Sort by similarity and get top 3 recommendations
    recommendations = dataset.sort_values(by='similarity', ascending=False).head(3)
    
    # Update session state with the recommended drinks
    st.session_state['recommended_drinks'].extend(recommendations['name'].tolist())
    
    return recommendations

st.title('The Cocktail-Experiment')

liked_cocktails_input = st.text_input('Enter Cocktails You Like (comma-separated):', 'mojito, margarita, moscow mule')
liked_cocktails = [cocktail.strip().lower() for cocktail in liked_cocktails_input.split(',')]

if st.button('Get Recommendations'):
    # Vectorize ingredients in the dataset
    vectorized_cocktails = vectorize_ingredients(space_cocktail)
    
    # Get recommendations based on the liked cocktails
    recommendations = get_recommendations(liked_cocktails, vectorized_cocktails)

    if not recommendations.empty:
        st.write("Top 3 recommendations based on your preferences:")
        for _, row in recommendations.iterrows():
            st.write(f"Recommended Drink: {row['name']}")
            ingredients = ', '.join([str(ingredient) for ingredient in filter(None, [
                row['ingredient-1'], row['ingredient-2'], row['ingredient-3'],
                row['ingredient-4'], row['ingredient-5'], row['ingredient-6']
            ])])
            st.write(f"Ingredients: {ingredients}")
            st.write(f"Similarity: {row['similarity']:.2f}")






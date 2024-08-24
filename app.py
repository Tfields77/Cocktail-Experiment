import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the cocktail dataset
space_cocktail = pd.read_csv('space_cocktail.csv')

# Initialize a session state to keep track of recommended drinks
if 'recommended_drinks' not in st.session_state:
    st.session_state['recommended_drinks'] = []

def parse_menu_text(menu_text):
    menu_items = []
    for item in menu_text.split('\n'):
        item = item.strip()
        if item and item[0].isalpha():  # Only consider lines that start with a letter
            menu_items.append(item)
    return menu_items

def create_menu_dataframe(menu_items):
    menu_data = {
        'name': [],
        'ingredient-1': [],
        'ingredient-2': [],
        'ingredient-3': [],
        'ingredient-4': [],
        'ingredient-5': [],
        'ingredient-6': []
    }

    for item in menu_items:
        parts = item.split(',')
        name = parts[0]
        ingredients = parts[1:]
        
        menu_data['name'].append(name)
        for i in range(6):
            if i < len(ingredients):
                menu_data[f'ingredient-{i+1}'].append(ingredients[i].strip())
            else:
                menu_data[f'ingredient-{i+1}'].append(None)
    
    return pd.DataFrame(menu_data)

def vectorize_ingredients(data):
    data['recipe_vector'] = data.apply(lambda row: ' '.join([str(ingredient) for ingredient in filter(None, [
        row['ingredient-1'], row['ingredient-2'], row['ingredient-3'],
        row['ingredient-4'], row['ingredient-5'], row['ingredient-6']
    ])]), axis=1)
    return data

def get_recommendations(input_ingredients, dataset):
    # Remove already recommended drinks
    dataset = dataset[~dataset['name'].isin(st.session_state['recommended_drinks'])]
    
    if dataset.empty:
        st.warning("No new drinks to recommend based on the current session.")
        return pd.DataFrame()  # Return an empty DataFrame if all drinks have been recommended
    
    # Vectorize the entire dataset and the input ingredients
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(dataset['recipe_vector'])
    
    # Vectorize the input ingredients (converting the list to a single string)
    input_vector = vectorizer.transform([' '.join(input_ingredients)])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(vectorized_data, input_vector)
    
    # Add similarity scores to the dataset
    dataset['similarity'] = cosine_sim.flatten()
    
    # Sort by similarity and get top 3 recommendations
    recommendations = dataset.sort_values(by='similarity', ascending=False).head(3)
    
    # Update session state with the recommended drinks
    st.session_state['recommended_drinks'].extend(recommendations['name'].tolist())
    
    return recommendations

st.title('The Cocktail-Experiment')

liked_ingredients_input = st.text_input('Enter Ingredients You Like (comma-separated):', 'lime, mint, rum')
liked_ingredients = [ingredient.strip().lower() for ingredient in liked_ingredients_input.split(',')]

uploaded_image = st.file_uploader("Upload a Picture of a Menu", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'heic', 'tif'])

if st.button('Submit'):
    if uploaded_image is not None:
        try:
            with st.spinner('Processing the image...'):
                image = Image.open(uploaded_image)
                image = image.resize((800, 800))  # Resize to 800x800 pixels
                
                menu_text = pytesseract.image_to_string(image)
                st.image(image, caption='Uploaded Menu')
                st.write("Extracted text from the menu:")
                st.write(menu_text)
                
                menu_items = parse_menu_text(menu_text)
                menu_df = create_menu_dataframe(menu_items)
                st.write("Parsed Menu DataFrame:")
                st.write(menu_df)

                # Vectorize ingredients in the dataset
                vectorized_cocktails = vectorize_ingredients(space_cocktail)
                
                # Get recommendations based on the input ingredients
                recommendations = get_recommendations(liked_ingredients, vectorized_cocktails)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        # If no image is uploaded, recommend based on the liked ingredients
        vectorized_cocktails = vectorize_ingredients(space_cocktail)
        recommendations = get_recommendations(liked_ingredients, vectorized_cocktails)

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







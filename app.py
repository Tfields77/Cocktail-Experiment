import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the cocktail dataset
space_cocktail = pd.read_csv('space_cocktail.csv')

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

def filter_dataset(dataset, liked_cocktails):
    return dataset[dataset['name'].str.lower().isin(liked_cocktails)]

def vectorize_ingredients(data):
    data['recipe_vector'] = data.apply(lambda row: ' '.join([str(ingredient) for ingredient in filter(None, [
        row['ingredient-1'], row['ingredient-2'], row['ingredient-3'],
        row['ingredient-4'], row['ingredient-5'], row['ingredient-6']
    ])]), axis=1)
    return data

def get_recommendations(menu_df, liked_ingredients_vectorized):
    count = CountVectorizer().fit_transform(menu_df['recipe_vector'])
    cosine_sim = cosine_similarity(count, liked_ingredients_vectorized)
    
    menu_df['similarity'] = cosine_sim.mean(axis=1)
    return menu_df.sort_values(by='similarity', ascending=False).head(3)

st.title('The Cocktail-Experiment')

liked_cocktails = st.text_input('Enter Cocktails You Like (comma-separated):', 'mojito, margarita, moscow mule, old-fashioned, manhattan, negroni')
liked_cocktails = [cocktail.strip().lower() for cocktail in liked_cocktails.split(',')]

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

                filtered_cocktails = filter_dataset(space_cocktail, liked_cocktails)
                vectorized_cocktails = vectorize_ingredients(filtered_cocktails)
                liked_ingredients_vectorized = CountVectorizer().fit_transform(vectorized_cocktails['recipe_vector'])
                
                recommendations = get_recommendations(menu_df, liked_ingredients_vectorized)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        # If no image is uploaded, just use the liked_cocktails for recommendations
        filtered_cocktails = filter_dataset(space_cocktail, liked_cocktails)
        vectorized_cocktails = vectorize_ingredients(filtered_cocktails)
        recommendations = filtered_cocktails.head(3)

    st.write("Top 3 recommendations based on your preferences:")
    for _, row in recommendations.iterrows():
        st.write(f"Menu Item: {row['name']}")
        ingredients = ', '.join([str(ingredient) for ingredient in filter(None, [
            row['ingredient-1'], row['ingredient-2'], row['ingredient-3'],
            row['ingredient-4'], row['ingredient-5'], row['ingredient-6']
        ])])
        st.write(f"Ingredients: {ingredients}")







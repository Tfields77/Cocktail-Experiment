import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pytesseract
from PIL import Image

# Define your function to parse menu from image
def parse_menu(image_path):
    menu_text = pytesseract.image_to_string(Image.open(image_path))
    menu_items = [item.strip() for item in menu_text.split('\n') if item.strip()]
    
    menu_data = {
        "name": [],
        "ingredient-1": [],
        "ingredient-2": [],
        "ingredient-3": [],
        "ingredient-4": [],
        "ingredient-5": [],
        "ingredient-6": [],
        "ingredient-7": []
    }
    
    for item in menu_items:
        if ',' in item:
            parts = item.split(',')
            menu_data["name"].append(parts[0].strip())
            ingredients = parts[1:]
            for i in range(1, 8):
                if i <= len(ingredients):
                    menu_data[f"ingredient-{i}"].append(ingredients[i-1].strip())
                else:
                    menu_data[f"ingredient-{i}"].append(None)
        else:
            menu_data["name"].append(item)
            for i in range(1, 8):
                menu_data[f"ingredient-{i}"].append(None)

    return pd.DataFrame(menu_data)

# Define function to compute cosine similarity
def compute_similarity(user_cocktails, menu_df):
    user_ingredients = ' '.join(user_cocktails)
    
    menu_df['combined_ingredients'] = menu_df[
        [f'ingredient-{i}' for i in range(1, 8)]
    ].fillna('').agg(' '.join, axis=1)
    
    vectorizer = CountVectorizer().fit_transform([user_ingredients] + menu_df['combined_ingredients'].tolist())
    vectors = vectorizer.toarray()
    
    cosine_matrix = cosine_similarity(vectors)
    
    menu_df['similarity'] = cosine_matrix[0, 1:]
    return menu_df[['name', 'similarity']].sort_values(by='similarity', ascending=False).head(3)

# Streamlit app
st.title('The Cocktail-Experiment')
st.write('Enter the cocktails you like and upload a picture of the menu.')

cocktails_input = st.text_input('Enter cocktails you like (comma-separated):', 'mojito, margarita, moscow mule, old-fashioned, manhattan, negroni')
uploaded_image = st.file_uploader('Upload a Picture of a Menu', type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'heic'])

if st.button('Submit'):
    if uploaded_image is not None:
        cocktails_list = [x.strip() for x in cocktails_input.split(',')]
        
        with open("/mnt/data/menu_image.png", "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        parsed_menu_df = parse_menu("/mnt/data/menu_image.png")
        st.write('Parsed Menu DataFrame:')
        st.dataframe(parsed_menu_df)
        
        recommendations = compute_similarity(cocktails_list, parsed_menu_df)
        
        st.write('Top 3 recommendations based on your preferences:')
        for index, row in recommendations.iterrows():
            st.write(f"Menu Item: {row['name']}")
            st.write(f"Similarity Score: {row['similarity']:.2f}")
    else:
        st.write("Please upload a menu image.")







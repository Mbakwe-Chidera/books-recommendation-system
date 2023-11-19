import streamlit as st
import pandas as pd
import pickle
import requests

# Load books and similarity data
books = pickle.load(open('book_lists.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
book_list = books['title'].values

# Streamlit app
st.header('Books Recommender System')

# Create a dropdown to select a book
selected_book = st.selectbox('Select a Book', book_list)

# Function to fetch book information from Google Books API
def fetch_book_info(book_title):
    base_url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": f"{book_title}", "maxResults": 1}

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and data["items"]:
            book_info = data["items"][0]["volumeInfo"]
            return {
                "title": book_info.get("title", ""),
                "author": ", ".join(book_info.get("authors", [])),
                "image_url": book_info.get("imageLinks", {}).get("thumbnail", ""),
            }

    return None

# Fetch book information for the selected book
selected_book_info = fetch_book_info(selected_book)

# Display the selected book and its image
st.write(f"Selected Book: {selected_book}")
if selected_book_info:
    st.image(selected_book_info["image_url"], caption=selected_book_info["title"])
else:
    st.warning("Book cover not found.")

# Add the new function get_recommendations using Google Books API
def get_recommendations(selected_book, similarity, books):
    try:
        index = books[books['title'] == selected_book].index[0]
        distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
        recommend_books = []
        recommend_poster = []
        for i in distance[1:6]:
            book_title = books.iloc[i[0]]['title']
            book_info = fetch_book_info(book_title)
            if book_info:
                recommend_books.append(book_info['title'])
                recommend_poster.append(book_info['image_url'])
        return recommend_books, recommend_poster
    except IndexError:
        return [], []

# Display the recommended books when the button is clicked
if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_book, similarity, books)

    # Display the recommended books with images
    for i in range(len(recommendations[0])):
        st.text(recommendations[0][i])
        st.image(recommendations[1][i])

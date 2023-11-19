#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


# In[20]:


# Load audiobook data
books = pd.read_csv('books.csv')
books.head()


# In[3]:


books.title


# In[4]:


books.info()


# In[5]:


books.isnull().sum()


# In[6]:


books.duplicated().sum()


# In[7]:


books.describe().style.background_gradient(cmap="summer")


# ## DATA PREPROCESSING

# In[8]:


def clean_average_rating(value):
    try:
        return float(value)
    except ValueError:
        return None  # Set non-numeric values to None or NaN


# In[9]:


# Function to convert 'num_pages' to integer
def convert_to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None  # Set invalid values to None or NaN

# Convert 'num_pages' to integers
books['  num_pages'] = books['  num_pages'].apply(convert_to_int)


# In[10]:


#trim extra spaces
import re

# Function to preprocess titles
def preprocess_title(title):
    # Remove content within parentheses and extra spaces
    title = re.sub(r'\(.*\)', '', title).strip()
    return title

# Preprocess the titles in your dataset
books['title'] = books['title'].apply(preprocess_title)


# In[11]:


# Apply the cleaning function to the 'average_rating' column
books['average_rating'] = books['average_rating'].apply(clean_average_rating)


# In[25]:


# Function to convert various date formats to datetime
def convert_to_datetime(value):
    try:
        return pd.to_datetime(value, errors='coerce')  # Use 'coerce' to handle invalid dates
    except (ValueError, TypeError):
        return None  # Invalid values can be set to None or NaN

# Standardize dates
books['publication_date'] = books['publication_date'].apply(convert_to_datetime)

# Now, 'publication_date' should contain standardized dates with some invalid dates set to NaN


# In[13]:


# Text data preprocessing (e.g., lowercase, tokenization)
books['title'] = books['title'].str.lower()
books['authors'] = books['authors'].str.lower()


# ## DATA VISUALIZATION

# In[14]:


# Create a histogram for the 'average_rating' variable
plt.hist(books['average_rating'], bins=20)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()


# In[15]:


# Distribution of authors
top_authors = books['authors'].value_counts().head(10)
sns.barplot(x=top_authors, y=top_authors.index)
plt.title('Top 10 Authors by Number of Books')
plt.xlabel('Number of Books')
plt.ylabel('Author')
plt.show()


# In[16]:


movie_ratings = books['title'].value_counts().head(10)

# Plotting the top 10 most rated movies

sns.barplot(x=movie_ratings.values, y=movie_ratings.index)
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.title('Top 10 Most Rated Movies')
plt.show()


# In[17]:


# Create a histogram for the 'average_rating' variable
plt.hist(books['  num_pages'], bins=20)
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.show()


# In[18]:


# Count and visualize the distribution of 'authors'
authors_counts = books['authors'].value_counts()
authors_counts


# In[26]:


# Task 5: Time Series Analysis
# Convert 'publication_date' to a datetime type
books['publication_date'] = pd.to_datetime(books['publication_date'])
# Extract year and create a line plot
books['publication_year'] = books['publication_date'].dt.year
year_counts = books['publication_year'].value_counts().sort_index()
year_counts.plot(kind='line')
plt.xlabel('Year')
plt.ylabel('Number of Books')
plt.show()


# In[21]:


# Task 6: Correlation Analysis
correlation_matrix = books.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[27]:


# Calculate book age based on 'publication_date'
current_year = 2023
books['book_age'] = current_year - books['publication_year']
books['book_age'].head()


# ## BUILDING RECOMMENDER SYSTEM

# In[52]:


# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=11000, stop_words='english')

# Fit and transform the 'title' and 'authors' columns
tfidf_matrix_title = tfidf_vectorizer.fit_transform(books['title'])
tfidf_matrix_authors = tfidf_vectorizer.fit_transform(books['authors'])


# For content-based recommendation, you need to create item profiles from the textual data, such as 'title' and 'authors.' One common approach is to use TF-IDF vectorization to convert the text data into numerical features.
# 
# Step 3: Calculate Similarity Scores
# 
# You can calculate the cosine similarity between items (books) based on their TF-IDF vectors. This will allow you to determine how similar or dissimilar two books are in terms of their titles and authors:

# In[29]:


# Calculate cosine similarity for titles
cosine_sim = linear_kernel(tfidf_matrix_title, tfidf_matrix_title)

# Calculate cosine similarity for authors
cosine_sim_authors = linear_kernel(tfidf_matrix_authors, tfidf_matrix_authors)


# ## Step 4: Generate Recommendations
# 
# Now that you have calculated similarity scores for titles and authors, you can use these scores to generate recommendations for a given book. For example, to get book recommendations based on a selected book:

# In[30]:


import pandas as pd

def get_recommendations(title, cosine_sim=cosine_sim, books=books):
    try:
        # Convert the input title to lowercase
        title = title.lower()

        # Convert the titles in the dataset to lowercase
        books['title'] = books['title'].str.lower()

        # Get the index of the book with the given title
        idx = books[books['title'] == title].index[0]

        # Get the pairwise similarity scores with all books
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar books (you can change the number)
        sim_scores = sim_scores[1:11]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Create a DataFrame with the recommended book titles
        recommended_books = books['title'].iloc[book_indices]

        # Return the DataFrame
        return pd.DataFrame({'Recommended Books': recommended_books})
    except IndexError:
        return pd.DataFrame({'Recommended Books': ["Book not found in the dataset."]})

# Example usage:
recommended_books = get_recommendations("You Bright and Risen Angels")
recommended_books


# In[31]:


books.title


# In[32]:



def get_recommendations(title, cosine_sim=cosine_sim, books=books):
    try:
        # Convert the input title to lowercase
        title = title.lower()

        # Convert the titles in the dataset to lowercase
        books['title'] = books['title'].str.lower()

        # Get the index of the book with the given title
        idx = books[books['title'] == title].index[0]

        # Get the pairwise similarity scores with all books
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 most similar books (you can change the number)
        sim_scores = sim_scores[1:11]

        # Convert 'average_rating' column to float
        books['average_rating'] = pd.to_numeric(books['average_rating'], errors='coerce')

        # Filter books by ratings (you can customize the rating threshold)
        rated_books = books[books['average_rating'] > 3.0]  # Example: Only consider books with a rating greater than 3.0

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Get recommended books that are also highly rated
        recommended_books = rated_books[rated_books.index.isin(book_indices)]

        # Sort the recommended books based on their ratings
        recommended_books = recommended_books.sort_values(by='average_rating', ascending=False)

        # Return the top recommended book titles and ratings in a DataFrame
        return recommended_books[['title', 'average_rating']]
    except IndexError:
        print("Books not in dataset.")
        return pd.DataFrame()

# Example usage:
recommended_books = get_recommendations("Las aventuras de Tom Sawyer")
recommended_books


#  This function will return a list of recommended book titles along with their associated ratings, sorted by ratings in descending order.

# ## SEARCH ENGINE

# In[33]:


engine = books
engine["rating"] = pd.to_numeric(engine['average_rating'],errors='coerce')
engine["Re_title"] = engine["title"].str.replace("[^a-zA-Z0-9]"," ",regex=True)
engine.to_json("books_engine.json")
engine.head()


# this code prepares the engine DataFrame for further analysis by converting the "average_rating" column to a numeric format and cleaning up the "title" column for text-based operations.

# In[34]:


Vectorize = TfidfVectorizer()
Tfvect = Vectorize.fit_transform(engine["Re_title"])
def search(Query, Vectorize):
    sub_match = re.sub("[^a-zA-Z0-9]"," ", Query.lower())
    Query_vec = Vectorize.transform([sub_match])
    Similarity = cosine_similarity(Query_vec, Tfvect).flatten()
    indexes = np.argpartition(Similarity, -10)[-10:]
    finall = engine.iloc[indexes]
    finall = finall.sort_values("average_rating", ascending = False)
    return finall.head(5)
search("tom",Vectorize)


# this function is to provide book recommendations based on a search query, and it ensures that the returned books are both relevant to the query and highly rated.

# In[35]:


from sklearn.metrics.pairwise import cosine_similarity

def search_by_author(Query, author, Vectorize, engine, num_books=10):
    sub_match = re.sub("[^a-zA-Z0-9]", " ", Query.lower())
    Query_vec = Vectorize.transform([sub_match])

    # Convert author names to lowercase for case-insensitive matching
    author = author.lower()
    
    # Filter books by the same author (case-insensitive)
    same_author_books = engine[engine['authors'].str.lower() == author]

    print("Number of books by the same author:", len(same_author_books))

    # Check if there are books by the author
    if same_author_books.empty:
        print("No books by the specified author.")
        return pd.DataFrame()

    # Get the TF-IDF vectors for book titles by the same author
    Tfvect_same_author = Vectorize.transform(same_author_books["Re_title"])

    Similarity = cosine_similarity(Query_vec, Tfvect_same_author).flatten()

    # Ensure num_books is within the range of available books
    num_books = min(num_books, len(same_author_books))

    indexes = Similarity.argsort()[-num_books:][::-1]
    recommended_books = same_author_books.iloc[indexes]

    # Sort the results by average rating in descending order
    recommended_books = recommended_books.sort_values("average_rating", ascending=False)

    return recommended_books.head(5)

# Example usage: Search for books by the same author as "Tom Sawyer" with the query "Tom"
author = "Mark Twain"  # Replace with the author of "Tom Sawyer" (case-insensitive)
search_results = search_by_author("Tom", author, Vectorize, engine, num_books=10)
search_results


# the above code allows you to specify the number of books by the same author you want to retrieve using the num_books parameter. The code ensures that num_books doesn't exceed the available number of books by the same author.

# In[38]:


get_recommendations('tom ford')


# In[36]:


import requests

#Google Custom Search JSON API key
API_KEY = 'AIzaSyAuC0qYvedzdXGDyHW7wi0Ix5WCB3joSg0'

def perform_web_search(query):
    base_url = 'https://www.googleapis.com/customsearch/v1'
    cx = '705412536359f4cac'  #Custom Search Engine ID

    params = {
        'key': API_KEY,
        'cx': cx,
        'q': query,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    return data

while True:
    user_input = input("Enter your search query (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    search_results = perform_web_search(user_input)
    
    if 'items' in search_results:
        print("Search results:")
        for i, item in enumerate(search_results['items'], start=1):
            print(f"{i}. {item['title']}")
            print(item['link'])
    else:
        print("No search results found.")


# The script provides a simple command-line interface for performing web searches using the Google Custom Search API. It retrieves and displays search results based on the user's input query, and the user can continue searching until they decide to quit the program.

# In[53]:


from sklearn.metrics.pairwise import cosine_similarity
vector = tfidf_vectorizer.fit_transform(books['title'].values.astype('U')).toarray()
vector.shape


# In[51]:


similarity = cosine_similarity(tfidf_vectorizer)
similarity


# In[52]:




# In[39]:


import pickle


# In[54]:


pickle.dump(books, open('book_lists.pkl', 'wb'))


# In[55]:


pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[57]:


pickle.load(open('book_lists.pkl', 'rb'))


# In[ ]:





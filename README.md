# SYSTEM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Expanded dataset of movies
movies = pd.DataFrame({
    'title': [
        'The Matrix', 'Titanic', 'The Godfather', 'Star Wars', 'The Dark Knight', 
        'Inception', 'Interstellar', 'The Shawshank Redemption', 'Forrest Gump', 'Gladiator',
        'The Lion King', 'The Avengers', 'Jurassic Park', 'Pulp Fiction', 'The Lord of the Rings'
    ],
    'genre': [
        'Action, Sci-Fi', 'Drama, Romance', 'Crime, Drama', 'Action, Sci-Fi', 'Action, Crime', 
        'Action, Sci-Fi', 'Sci-Fi, Drama', 'Drama', 'Drama, Romance', 'Action, Drama', 
        'Animation, Adventure', 'Action, Sci-Fi', 'Adventure, Sci-Fi', 'Crime, Drama', 'Adventure, Fantasy'
    ],
    'description': [
        'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'Luke Skywalker joins forces with a Jedi knight, a cocky pilot, Chewbacca, and two droids to save the galaxy from the Empire\'s world-destroying battle station.',
        'When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'A thief who enters the dreams of others to steal secrets from their subconscious is given the task of planting an idea into the mind of a CEO.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Civil Rights Movement, and other historical events unfold from the perspective of an Alabama man with an incredible memory.',
        'A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.',
        'A young lion prince is cast out of his pride and must figure out how to claim his rightful place as king.',
        'Earth\'s mightiest heroes must come together and learn to fight as a team to stop a mischievous god from subjugating the Earth.',
        'During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok.',
        'The lives of two mob hit men, a boxer, a gangster\'s wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'A young hobbit sets out on an epic journey to destroy a powerful ring that could bring destruction to the world.'
    ]
})

# Function to recommend movies based on user input
def recommend_movies(user_input):
    # Convert the genres or descriptions into a list of strings to match
    movies['genre'] = movies['genre'].str.lower()
    movies['description'] = movies['description'].str.lower()

    # TfidfVectorizer to convert text (genres or descriptions) into numerical data
    vectorizer = TfidfVectorizer(stop_words='english')

    # Ask the user if they want genre or description-based recommendations
    print("\nHow would you like to get recommendations?")
    print("1. Based on Genre")
    print("2. Based on a Movie you like")
    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        # Ask the user for a genre preference
        genre_preference = input("Enter your preferred genre (e.g., Action, Sci-Fi): ").lower()

        # Filter the dataset based on the preferred genre
        genre_filtered = movies[movies['genre'].str.contains(genre_preference)]
        if not genre_filtered.empty:
            print(f"\nYou're interested in {genre_preference} movies. Here are some recommendations:")
            tfidf_matrix = vectorizer.fit_transform(genre_filtered['description'])
        else:
            print("Sorry, we couldn't find any movies for that genre. Please try a different one.")
            return
    else:
        # Ask the user for a movie title
        movie_title = input("Enter a movie you like (e.g., The Matrix): ").lower()
        movie_match = movies[movies['title'].str.lower() == movie_title]

        if movie_match.empty:
            print("Sorry, we couldn't find that movie. Please try again.")
            return
        
        print(f"\nYou're interested in movies similar to '{movie_title.capitalize()}'. Here are some recommendations:")
        tfidf_matrix = vectorizer.fit_transform(movies['description'])

    # Compute the cosine similarity between the input movie and the rest
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get movie recommendations
    movie_idx = movie_match.index[0] if not movie_match.empty else 0  # get the index of the movie the user liked
    similar_movies = cosine_sim[movie_idx].argsort()[-6:][::-1]  # exclude the first item as it's the same movie

    print("\nRecommended Movies:")
    for idx in similar_movies[1:]:  # skip the first index (which is the input movie)
        print(f"- {movies['title'].iloc[idx]}")

    # Ask user if they liked the recommendations
    feedback = input("\nDid you like the recommendations? (yes/no): ").lower()
    if feedback == 'no':
        print("Okay! Let's try something else.")
        recommend_movies(user_input)  # Recursively call the function for a new recommendation
    else:
        print("Great! Enjoy the movies!")

# Start the chatbot
print("Welcome to the movie recommendation system!")
recommend_movies('')


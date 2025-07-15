from flask import Flask, render_template, request, jsonify
from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

dataset = load_dataset("rohitvaddepalli/TeluguMovies", split="train")
df = pd.DataFrame(dataset)
df['Overview'] = df['Overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Movie']).drop_duplicates()
movie_list = df['Movie'].tolist()

def recommend(title, topn=5):
    if not isinstance(title, str) or title not in indices:
        return []
    idx = indices[title]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:topn+1]
    return df['Movie'].iloc[[i[0] for i in scores]].tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    recs = []
    selected_movie = ""
    if request.method == "POST":
        selected_movie = request.form.get("movie")
        recs = recommend(selected_movie)
    return render_template("index.html", recommendations=recs, selected_movie=selected_movie)

@app.route("/search")
def search():
    query = request.args.get("q", "").lower()
    suggestions = [movie for movie in movie_list if query in movie.lower()]
    return jsonify(suggestions[:10])

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask
from flask import url_for, render_template, request, redirect
from collections import Counter
import numpy as np

app = Flask(__name__)


def read_file():
    with open('Lemmas.txt', 'r', encoding='utf-8') as f:
        lemmas = f.read()
    return lemmas


def get_df(lemmas):
    DF = {}
    N = len(lemmas)
    for i in range(N):
        tokens = lemmas[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])

    return DF


def get_tf_idf(DF):
    doc = 0

    tf_idf = {}

    for i in range(N):

        tokens = lemmas[i]

        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / words_count
            df = doc_freq(token)
            idf = np.log((N + 1) / (df + 1))

            tf_idf[doc, token] = tf * idf

        doc += 1
    return tf_idf


def preprocess_query(query):
    query = query.lower()
    query = query.strip('!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n')

    return query


def get_tokens(query):
    preprocessed_query = preprocess_query(query)
    tokens = str(preprocessed_query).split()

    return tokens


def matching_score(k, tokens, tf_idf):
    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    results = []

    for i in query_weights[:10]:
        results.append(i[0])

    return results


@app.route('/')
def index():
    urls = {'main_page': url_for('index'),
            'results_data': url_for('show_results')}
    return render_template('index.html', urls=urls)


@app.route('/results')
def show_results():
    input_query = request.args['query']
    input_query = preprocess_query(input_query)
    tokens = get_tokens(input_query)
    lemmas = read_file()
    DF = get_df(lemmas)
    tf_idf = get_tf_idf(DF)
    results = matching_score(10, tokens, tf_idf)

    return render_template('results.html', query=input_query, results=results, tokens=tokens)


if __name__ == '__main__':
    app.run(debug=True)
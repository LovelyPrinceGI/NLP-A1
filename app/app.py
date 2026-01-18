# app.py
# Run this using `python app.py`

from flask import Flask, request, render_template_string
from search_backend import search_topk  # แบบใหม่มี argument model_name

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
    <title>Simple Embedding Search</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        input[type=text] { width: 400px; padding: 8px; }
        select { padding: 6px; margin-left: 8px; }
        .result { margin-top: 10px; padding: 8px; border-bottom: 1px solid #ddd; }
        .score { color: #888; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Embedding-based Search</h1>
    <form method="GET">
        <input type="text" name="q" placeholder="Enter your query"
               value="{{ query|default('') }}" />

        <select name="model">
            <option value="glove" {% if model == 'glove' %}selected{% endif %}>
                GloVe
            </option>
            <option value="skipgram_ns" {% if model == 'skipgram_ns' %}selected{% endif %}>
                Skip-gram (NS)
            </option>
        </select>

        <button type="submit">Search</button>
    </form>

    {% if query %}
        <h2>Results for "{{ query }}" (model: {{ model }})</h2>
        {% if results %}
            {% for r in results %}
                <div class="result">
                    <div>{{ r.text }}</div>
                    <div class="score">score: {{ "%.4f"|format(r.score) }}</div>
                </div>
            {% endfor %}
        {% else %}
            <p>No results (no in-vocabulary words in query).</p>
        {% endif %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    query = request.args.get("q", "").strip()
    model_name = request.args.get("model", "glove")  # default = GloVe
    results = []
    if query:
        results = search_topk(query, k=10, model_name=model_name)
    return render_template_string(HTML, query=query, results=results, model=model_name)

if __name__ == "__main__":
    app.run(debug=True)

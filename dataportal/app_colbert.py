from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
from .panosc_portal import PanoscPortalColbert

app = Flask(__name__)
CORS(app)

documents_path = 'data/panson_docs_summaries.json' # Ideally this would be in a config file or a database, but for simplicity and demonstartion i'll keep it here. 
portal = PanoscPortalColbert(documents_path=documents_path)

@app.route('/')
def home():
    return render_template('search.html')


@app.route('/index', methods=['POST'])
def index():
    documents = request.json
    portal.index_documents(documents)
    return jsonify({"message": "Documents indexed successfully"}), 200


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = portal.search(query)
        if not results:
            return jsonify({"error": "No results found"}), 404

        print(f"Number of results: {len(results)}")
        # limit the number of results returned
        # results = results[:10]

        return jsonify(results)  # Return the list of results as JSON

    except Exception as e:
        print(f"Error occurred during search: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

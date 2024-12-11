from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import requests
import json

app = Flask(__name__)
CORS(app) 

class PanoscPortal:
    def __init__(self):
        self.url = "https://federated.panosc.ess.eu/api/Documents"

    def search(self, query: str):
        params = self.prepare_query(query)
        response = requests.get(self.url, params=params)
        if response.status_code == 200:
            return response.json()  
        else:
            return {"error": f"Request failed with status code {response.status_code}"}

    def prepare_query(self, query: str):
        filter_param = {
            "query": query,
            "limit": 50
        }
        params = {
            "filter": json.dumps(filter_param)
        }
        return params

# Define the route for searching
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    portal = PanoscPortal()
    results = portal.search(query)
    return jsonify(results)  

if __name__ == '__main__':
    app.run(debug=True)

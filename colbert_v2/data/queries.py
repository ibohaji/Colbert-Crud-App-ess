# data/queries.py
class Queries:
    def __init__(self):
        self.queries = []

    def add_query(self, query):
        self.queries.append(query)

    def get_queries(self):
        return self.queries

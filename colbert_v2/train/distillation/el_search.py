import logging
import subprocess
import os
import time
import requests
from elasticsearch import Elasticsearch, helpers
from contextlib import ContextDecorator
import re 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EsSearcher(ContextDecorator):
    def __enter__(self):
        """This is called when entering the context (when Elasticsearch starts)."""
        try:
            self.es_process = self.start_elasticsearch()
            host, port = self.get_es_binding()
            self.wait_for_elasticsearch(host, port)
            self.es = Elasticsearch(f"http://{host}:{port}", timeout=30)
            self.index_name = 'documents'
            logger.info(f"Connected to Elasticsearch at {host}")

            # Delete the index if it exists
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index '{self.index_name}'")

            return self  # Allow chaining of methods
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            raise e


    def __exit__(self, exc_type, exc_val, exc_tb):
        """This is called when leaving the context (cleanup/termination)."""
        self.terminate_es()
        if exc_type:
            logger.error(f"Exception occurred: {exc_type}, {exc_val}")
            return False  # Propagate the exception
        return True

    def retrieve(self, index, query_text, positive_doc_id, num_results=10):
         search_body = {
            "size": num_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["title", "text"]                        }
                    },
                    "must_not": {
                        "ids": {
                            "values": [positive_doc_id]
                        }
                    }
                }
            }
        }

         response = self.es.search(index=self.index_name, body=search_body)
         return response['hits']['hits']


    def index_documents(self, collection_data):
        """Indexes documents into Elasticsearch using the bulk helper."""
        # Define the mapping for the index
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"}
                }
            }
        }

        try:
            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index '{self.index_name}' with mapping.")
        except Exception as e:
            logger.error(f"Error creating index '{self.index_name}': {e}")
            raise e

        # Prepare actions for bulk indexing
        actions = [
            {
                "_index": self.index_name,
                "_id": doc['_id'],
                "_source": {
                    "title": doc['title'],
                    "text": doc['text']
                }
            }
            for doc in collection_data
        ]

        try:
            success, failed = helpers.bulk(
                self.es,
                actions,
                refresh=True,
                raise_on_error=False,
                stats_only=False
            )
            logger.info(f"Successfully indexed {success} documents.")
            if failed:
                logger.warning(f"Failed to index {len(failed)} documents.")
                for failure in failed:
                    logger.warning(f"Failed document: {failure}")
        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            raise e
        


    def start_elasticsearch(self):
        print("Starting Elasticsearch...")
        
        stdout_log = open('es_stdout.log', 'w')
        stderr_log = open('es_stderr.log', 'w')

        es_process = subprocess.Popen(
            ['/zhome/3a/7/145702/elasticsearch-8.5.0/bin/elasticsearch'],
            stdout=stdout_log,
            stderr=stderr_log
        )
        logger.info("Elasticsearch started.")
        logger.info(f"PID: {es_process.pid}")

        return es_process
    
    def extract_es_binding_from_logs(self):
   
        logger.info("Extracting Elasticsearch bindings from logs...")
        while True:
            output = self.es_process.stdout.readline()  

            if output == '' and self.es_process.poll() is not None:
                logger.error("Elasticsearch process finished before detecting binding.")
                break

            # Parse the logs for binding information
            match = re.search(r"publish_address \{([0-9.:]+)\}", output)
            if match:
                bound_ip = match.group(1)
                logger.info(f"Elasticsearch is bound to {bound_ip}")
                return bound_ip.split(':')[0], bound_ip.split(':')[1]

            # For debugging: print every log line if needed
            logger.debug(f"Elasticsearch log: {output.strip()}")
        
        logger.error("Failed to extract Elasticsearch binding.")
        return None, None


    def wait_for_elasticsearch(self, host, port, max_retries=5):
        retries = 0
        logger.info(f"Waiting for Elasticsearch to be available at {host}:{port}...")

        while retries < max_retries:
            try:
                # Try to connect to Elasticsearch
                response = requests.get(f"http://{host}:{port}")
                if response.status_code == 200:
                    logger.info("Elasticsearch is ready!")
                    return
            except requests.exceptions.ConnectionError:
                retries += 1
                logger.warning(f"Elasticsearch not ready, retrying... ({retries}/{max_retries})")
                time.sleep(5)

        raise ConnectionError(f"Failed to connect to Elasticsearch after {max_retries} attempts.")
    

    def get_host_ip(self):
        host_ip = subprocess.getoutput("hostname -I").split()[0]
        logger.info(f"Detected host IP: {host_ip}")
        return host_ip


    def get_es_binding(self):
        logger.info("Checking Elasticsearch bindings...")

        try:
            output = subprocess.getoutput("netstat -tuln | grep :9200")
            logger.info(f"netstat output: {output}")
            if output:
                parts = output.split()
                bound_ip, bound_port = parts[3].split(':')
                logger.info(f"Elasticsearch is bound to {bound_ip}:{bound_port}")
                return bound_ip, bound_port
            else:
                raise Exception("Failed to detect Elasticsearch binding via netstat.")
        except Exception as e:
            logger.error(f"Error detecting Elasticsearch binding: {e}")
            raise e


    
    def terminate_es(self):
        if hasattr(self, 'es_process') and self.es_process:
            logger.info("Terminating Elasticsearch...")
            self.es_process.terminate()
            self.es_process.wait()
            logger.info("Elasticsearch terminated.")




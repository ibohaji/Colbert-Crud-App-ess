
# ColBERT CRUD Application

## About the Project
[Detailed description of the project yet to come]

This project implements a CRUD (Create, Read, Update, Delete) application using ColBERT (Contextualized Late Interaction over BERT) for efficient document retrieval and search. The application includes a data portal interface for easy interaction with the ColBERT system.

### Key Features
- Document indexing and retrieval using ColBERT
- Web interface for searching documents
- CRUD operations for document management
- Training and evaluation of ColBERT models

## Prerequisites

### System Requirements
- Unix/Linux environment (recommended)
  - Windows users: Consider using WSL or Conda environment
- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)

### Environment Setup

1. Clone the repository:
```bash
git clone git@github.com:ibohaji/Colbert-Crud-App-ess.git
cd Colbert-Crud-App-ess
```

2. Create and activate a virtual environment:
```bash
# Linux/Unix
python -m venv myenv
source myenv/bin/activate

# Windows (if not using WSL)
# Consider using Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Start the Data Portal
```bash
python -m dataportal.app_colbert
```
The portal will be available at `http://localhost:5000`

### Adding Documents to the Portal

1. Prepare your documents in JSON format:
```json
{
    "doc_id": {
        "title": "Document Title",
        "text": "Document content..."
    }
}
```

2. Use the API endpoint:
```bash
curl -X POST http://localhost:5000/index \
     -H "Content-Type: application/json" \
     -d @your_documents.json
```


## Acknowledgments & Credits

This project is built using ColBERTv2, an efficient and effective neural search engine:

- Original Paper: ["ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"](https://arxiv.org/abs/2112.01488)
- Authors: Omar Khattab, Christopher Potts, and Matei Zaharia
- Official Repository: [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
- ColBERT-AI Library: [colbert-ai](https://github.com/stanford-futuredata/ColBERT)



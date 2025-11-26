
# HyDE-PDF-Retriever

A **PDF-based question answering system** using **Hypothetical Document Embeddings (HyDE)**, HuggingFace embeddings, and Groq-powered LLMs. This project allows you to query PDF documents, generate a hypothetical document answering your question, and retrieve the most relevant sections from your PDFs.

---

## Features

* Load and preprocess PDF documents with `PyPDFLoader`.
* Split PDF content into chunks with `RecursiveCharacterTextSplitter`.
* Clean text by replacing tabs with spaces.
* Generate embeddings using **HuggingFace sentence transformers**.
* Create a **FAISS vector store** for efficient similarity search.
* Generate a **hypothetical document** answering user queries using Groq LLMs (`ChatGroq`).
* Retrieve the most relevant PDF sections based on the generated document.

---

## Tech Stack

* **Python 3.11+**
* **LangChain** (Community + Core modules)
* **HuggingFace Transformers & Embeddings**
* **FAISS** (Vector store for fast similarity search)
* **Groq LLM** (`ChatGroq`)
* **dotenv** (Environment variable management)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/HyDE-PDF-Retriever.git
cd HyDE-PDF-Retriever
```

2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate   # Linux/macOS
env\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add your `.env` file for API keys and configuration:

```
# Example .env
GROQ_API_KEY=your_groq_api_key
```

---

## Usage

1. Place your PDF file in the `data/` folder. Example:

```python
path = "data/Understanding_Climate_Change.pdf"
```

2. Initialize the HyDE retriever:

```python
from hyde_retriever import HyDERetriever

retriever = HyDERetriever(path)
```

3. Query your PDF and get relevant content:

```python
test_query = "What is the main cause of climate change?"
results, hypothetical_doc = retriever.retrieve(test_query)

# Display results
print("Hypothetical doc:\n", hypothetical_doc)
for i, doc in enumerate(results):
    print(f"Context {i+1}:\n", doc.page_content)
```

---

## How It Works

1. **PDF Loading & Splitting:**
   PDF content is loaded using `PyPDFLoader` and split into manageable chunks.

2. **Text Cleaning:**
   Tabs (`\t`) are replaced with spaces for clean processing.

3. **Embedding Creation:**
   Each chunk is embedded using **HuggingFace's MiniLM-L6-v2** model.

4. **Vector Store:**
   Chunks are indexed in **FAISS** for fast similarity search.

5. **HyDE Hypothetical Document Generation:**

   * Given a query, the system generates a detailed hypothetical document using `ChatGroq`.
   * This hypothetical document is then used to search for the most relevant sections in the PDF.

6. **Retrieval:**
   Returns both the hypothetical answer and the top-K most relevant PDF chunks.

---

## Example Output

**Query:** `"What is the main cause of climate change?"`

**Hypothetical Document:**

```
[Text of the generated document, wrapped at 120 characters]
```

**Top 3 Relevant Sections from PDF:**

```
Context 1: ...
Context 2: ...
Context 3: ...
```

---

## Folder Structure

```
HyDE-PDF-Retriever/
│
├─ data/                    # PDF files
├─ hyde_retriever.py        # Main code
├─ requirements.txt
├─ README.md
└─ .env                     # Environment variables
```

---

## Future Improvements

* Support multiple PDF inputs and batch queries.
* Add GUI with **Streamlit** for interactive querying.
* Allow customizable embedding models and LLMs.
* Include caching for faster repeated queries.

---


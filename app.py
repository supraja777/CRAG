import os
from dotenv import load_dotenv
import textwrap

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.chat_models import ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_groq import ChatGroq


load_dotenv()

path = "data/Understanding_Climate_Change.pdf"

def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def encode_pdf(path, chunk_size = 1000, chunk_overlap = 200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len
    )

    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLm-L6-v2")

    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore

def show_context(context):
    """
    Display the contents of the provided context list
    """

    for i, c in enumerate(context):
        print(f"Context {i + 1} : ")
        print(c)
        print("\n")

def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)

# Define HyDe retriever class - creating vector store, generating hypothetical document and retrieving

class HyDERetriever:
    def __init__(self, files_path, chunk_size = 500, chunk_overlap = 100):
       
        self.llm = llm = ChatGroq(
            model="llama-3.3-70b-versatile"   
        )
        self.embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLm-L6-v2")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(files_path, chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)

        self.hyde_prompt = PromptTemplate(
            input_variables = ["query", "chunk_size"],
            template = """
            Given the question {query}, generate a hypothetical document that directly answers this question. 

            The document should be detailed and in-depth. The document should have exactly {chunk_size} characters."""
        )

        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query" : query, "chunk_size" : self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content
    
    def retrieve(self, query, k = 3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(hypothetical_doc, k = k)
        return similar_docs, hypothetical_doc
    
retriever = HyDERetriever(path)

test_query = "What is the main cause of climate change?"

results, hypothetical_doc = retriever.retrieve(test_query)

docs_content = [doc.page_content for doc in results]

print("Hypothetical doc \n")

print(text_wrap(hypothetical_doc) + "\n")
show_context(docs_content)




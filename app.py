import json
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Dict, Any, Tuple

import os
from dotenv import load_dotenv

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

vectorstore = encode_pdf(path)
llm = ChatGroq(model="llama-3.3-70b-versatile")

search = DuckDuckGoSearchRun()

# Define Retrieval Evaluator, Knowledge Refinement and Query Rewriter LLM chains

# Retrieval Evaluator

class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(description="The relevance score of the document to the query. The score should be between 0 and 1")

def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate(
        input_variables = ["query", "document"],
        template = """
        On a scale from 0 to 1, how relevant is the following document to the query? 
        Query: {query}
        Document: {document}
        Relevance score: 
        """
    )

    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)
    input_variables = {"query" : query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(description = "The document to extract key information from.")


# Knowledge Refinement

def knowledge_refinment(document: str) -> List[str]:
    prompt = PromptTemplate(
        input_variables = ["document"],
        template = """ Extract the key information from the following in bullet points: \n 
        {document}
        key points: 
        """
    )

    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)

    input_variables = {"document" : document}

    result = chain.invoke(input_variables).key_points

    return [point.strip() for point in result.split('\n') if point.strip()]


# Web Search Query Rewriter

class QueryRewriterInput(BaseModel):
    query: str = Field(description = "The query to rewrite")

def rewrite_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables = ["query"],
        template = """Rewrite the following query to make it more suitable for a web search: 
        {query} rewritten query: 
        """
    )

    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query" : query}
    return chain.invoke(input_variables).query.strip()

def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    try:
        # Attempt to parse json string
        results = json.loads(results_string)

        # Extract and return the title and link from each result
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        # Handle JSON decoding errors by returning an empty list
        print("Error parsing search results. Returning empty list")
        return []
    

def retrieve_documents(query: str, faiss_index: FAISS, k: int = 3) -> List[str]:
    """
    Retrieve documents based on a query using a FAISS index.

    Args:
        query (str): The query string to search for.
        faiss_index (FAISS): The FAISS index used for similarity search.
        k (int): The number of top documents to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of the retrieved document contents.
    """

    docs = faiss_index.similarity_search(query, k = k)
    return [doc.page_content for doc in docs]

def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    Evaluate the relevance of documents based on a query.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """

    return [retrieval_evaluator(query, doc) for doc in documents]

def perform_web_search(query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Perform a web search based on a query.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: 
            - A list of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """

    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinment(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources

def generate_response(query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
    """
    Generate a response to a query using knowledge and sources.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """

    response_prompt = PromptTemplate(
        input_variables = ["query", "knowledge", "sources"],
        template = "Based on the following knowledge, " \
        "answer the query. Include the sources with their links (if available)" \
        "at the end of your answer : Query : {query} " \
        "Knowledge: {knowledge}" \
        "Sources: {sources}" \
        "Answer: "
    )

    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }

    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content



# CRAG Process

def crag_process(query: str, faiss_index: FAISS) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.

    Args:
        query (str): The query string to process.
        faiss_index (FAISS): The FAISS index used for document retrieval.

    Returns:
        str: The generated response based on the query.
    """
    print(f"Processing query : {query}")

    # Retrieve and Evaluate documents
    retrieved_docs = retrieve_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)

    print(f"\nRetrieved {len(retrieved_docs)} documents")
    print(f"Evaluation scores: {eval_scores}")

    max_score = max(eval_scores)
    sources = []

    if max_score > 0.7:
        print("Action: Correct - Using retrieved documents")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        print("Action: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        print("Action: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        retrieved_knowledge = knowledge_refinment(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    print("Final Knowledge : ")
    print(final_knowledge)

    print("\n sources: ")
    for title, link in sources:
        print(f"{title} : {link}" if link else title)
    
    # Generate response
    print("\n Generating response .......")
    response = generate_response(query, final_knowledge, sources)

    print("\nResponse generated")
    return response

query = "What are the main causes of climate changes?"
result = crag_process(query, vectorstore)
print(f"Query: {query}")
print(f"Answer {result}")


query = "how did harry beat quirrell?"
result = crag_process(query, vectorstore)
print(f"Query: {query}")
print(f"Answer {result}")









    

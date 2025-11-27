# 🧠 **CRAG Pipeline — Corrective Retrieval-Augmented Generation**

### *A Smarter, More Reliable Alternative to Traditional RAG*

Retrieval-Augmented Generation (RAG) is widely used to enhance LLMs with external knowledge. However, **traditional RAG suffers from a major flaw**:

### ❌ **It blindly trusts retrieved documents—even when they are irrelevant.**

This leads to:

* Hallucinations
* Incorrect answers
* Over-reliance on possibly wrong retrieved chunks
* Poor performance on out-of-domain questions

To overcome this behavior, researchers introduced **CRAG — Corrective RAG**.

---

# 🚀 **What is CRAG?**

**CRAG (Corrective Retrieval-Augmented Generation)** is an improved retrieval pipeline designed to:

### ✔️ Validate retrieved documents

### ✔️ Detect incorrect retrieval

### ✔️ Trigger fallback actions

### ✔️ Combine multiple knowledge channels

### ✔️ Reduce hallucinations by forcing corrective behavior

In CRAG, retrieval is followed by a **relevance evaluation step**. Based on this score:

| Relevance Score      | Action                                       |
| -------------------- | -------------------------------------------- |
| **High (> 0.7)**     | Use the retrieved document                   |
| **Low (< 0.3)**      | Discard retrieval → switch to web search     |
| **Medium (0.3–0.7)** | Hybrid mode → combine retrieval + web search |

This makes CRAG far more robust and accurate.

---

# 🆚 **CRAG vs Traditional RAG**

## 🟦 **Traditional RAG**

* Retrieves top-k chunks
* Feeds them blindly to the LLM
* Assumes retrieval is always correct

**Problems:**

* If retrieval is irrelevant → LLM produces wrong answers
* Cannot handle out-of-scope queries
* Does not adapt dynamically
* High hallucination risk

---

## 🟩 **CRAG — Corrective RAG**

CRAG introduces a **corrective decision layer**, making it:

### ⭐ **More Accurate**

Irrelevant chunks are filtered via an LLM evaluator.

### ⭐ **More Adaptive**

If documents are irrelevant → it switches to **web search**.

### ⭐ **More Reliable**

Hybrid mode ensures a balanced mix of local + online knowledge.

### ⭐ **Less Hallucinatory**

Only high-confidence knowledge is allowed into final generation.

---

# 📊 **Pipeline Overview (CRAG Architecture)**

```
          ┌─────────────────────────┐
          │        User Query        │
          └─────────────┬───────────┘
                        ↓
              ┌──────────────────┐
              │ Retrieve Top-k    │
              │ (FAISS Vector DB) │
              └─────────┬────────┘
                        ↓
      ┌────────────────────────────────┐
      │ LLM-based Relevance Evaluator  │
      └─────────┬───────────┬────────┘
                │           │
     High Score │           │ Low Score
    (> 0.7)     │           │ (< 0.3)
                ↓           ↓
     Use PDF Doc│     Perform Web Search
         ↓       │           ↓
     ┌──────────────┐   ┌───────────────┐
     │ Knowledge     │   │ Web Knowledge │
     │ Refinement    │   │ Refinement    │
     └──────┬────────┘   └───────┬──────┘
            │                    │
            └────────┬───────────┘
                     ↓
          ┌──────────────────────────┐
          │   Response Generation    │
          └──────────────────────────┘
```

---

# 🧩 **Advantages of CRAG**

### 🟢 **1. High Accuracy**

Irrelevant chunks are filtered out before reaching the LLM.

### 🟢 **2. Scalable & Domain-Agnostic**

Works for:

* LLM apps
* PDF QA
* Web-assisted question answering
* Hybrid knowledge systems

### 🟢 **3. Reduced Hallucinations**

CRAG only uses **validated** or **trusted** knowledge.

### 🟢 **4. Fallback Mechanism**

If retrieval fails → automatic web search.

### 🟢 **5. Hybrid Reasoning**

CRAG combines:

* Vector DB knowledge
* Web knowledge
* LLM reasoning

…based on confidence.

### 🟢 **6. Better handling of out-of-domain queries**

Traditional RAG fails when query ≠ document domain.
CRAG performs a **dynamic correction**.

---

# 🧠 **CRAG Implementation**


✔ **PDF processing** with LangChain
✔ **Text splitting** (RecursiveCharacterSplitter)
✔ **Embeddings** (HuggingFace MiniLM)
✔ **FAISS Vectorstore**
✔ **Groq Llama 3.3 70B LLM**
✔ **Relevance scoring using structured output**
✔ **Knowledge refinement**
✔ **Query rewriting for web search**
✔ **Fallback logic**
✔ **Final answer generation with sources**

Below is a breakdown for each stage:

---

## 📥 1. **PDF Encoding**

* Load PDF using `PyPDFLoader`
* Split into chunks
* Remove weird tab characters
* Convert to embeddings using `all-MiniLM-L6-v2`
* Store vectors in FAISS

---

## 🔍 2. **Document Retrieval**

```
docs = faiss_index.similarity_search(query, k=3)
```

Retrieves top-k based on cosine similarity.

---

## 🧪 3. **Evaluation (CRAG Correction Layer)**

The evaluator uses:

```python
class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float
```

LLM decides a score between **0 and 1**.

---

## 🔄 4. **Decision Logic**

### If **score > 0.7**

→ Use retrieved document

### If **score < 0.3**

→ Web search (DuckDuckGo)

### If **0.3–0.7**

→ Combine retrieval + web search

This is the **core of CRAG**.

---

## 📝 5. **Knowledge Refinement**

Extracts bullet-point knowledge from documents or search results.

---

## 🌐 6. **Web Search + Query Rewriting**

If retrieval fails:

* Query → rewritten for search
* DuckDuckGo returns results
* Key information extracted

---

## 🧾 7. **Final Answer Generation**

Adds:

* context
* reasoning
* sources with links

---

# 🧪 **Sample Output (From Your Run)**

Your pipeline correctly detected:

### Query 1: *"What are the main causes of climate change?"*

* Retrieved documents relevant (score 0.8)
  → Uses PDF-based knowledge.

### Query 2: *"How did Harry beat Quirrell?"*

* Retrieval totally irrelevant (score 0.0)
  → CRAG switched to **web search**.
  → Extracted relevant story info.

This demonstrates CRAG working exactly as intended.

---

# 🏁 **Conclusion**

CRAG is a **superior evolution of RAG**, designed for reliability, correctness, and adaptive knowledge retrieval.

---
# RAG-based QA Test Case Assistant

A Retrieval Augmented Generation (RAG) system that enables QA engineers to query test cases using natural language, with comprehensive evaluation metrics including F1 score, precision, recall, MRR, NDCG, and cosine similarity.

## 📋 Table of Contents
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)

---

## 🏗️ System Architecture

### High-Level Overview

```
User Query → Embedding → Vector Search → Document Retrieval → LLM Generation → Answer
                ↓              ↓                ↓                    ↓
           Sentence      ChromaDB         Top-K Results      GPT-4 with Context
          Transformer
```

### Component Breakdown

#### 1. **Data Ingestion (`load_test_cases()`)** 
```python
def load_test_cases(self) -> None
```
**What it does:**
- Scans the `test_cases/` directory
- Loads all `.json` files (TC2260.json, TC2268.json, etc.)
- Parses JSON and stores in memory dictionary
- Handles encoding errors gracefully

**Why it matters:**
- Foundation for the entire system
- Validates data integrity before processing
- Provides error logging for debugging

---

#### 2. **Document Chunking (`chunk_test_case()`)** 
```python
def chunk_test_case(self, test_case: Dict, filename: str) -> List[Dict]
```
**What it does:**
Creates **4 semantic chunks** per test case:

1. **Overview Chunk**: Title + Summary + Description
2. **Steps Chunk**: All test steps sequentially
3. **Expected Results Chunk**: Expected outcomes
4. **Metadata Chunk**: Tags, priority, category, preconditions

**Example:**
```
Input: TC25977.json
Output:
  Chunk 1 (overview): "Title: Spanish Onboarding\nSummary: Tests Spanish language..."
  Chunk 2 (steps): "Test Steps:\n1. Navigate to signup\n2. Select Spanish..."
  Chunk 3 (expected): "Expected Results: User sees Spanish interface"
  Chunk 4 (metadata): "Tags: onboarding, spanish\nPriority: high"
```

**Why this strategy:**
- **Granular retrieval**: Find specific sections (e.g., just steps)
- **Better matching**: Query about "steps" retrieves step chunks
- **Context preservation**: Each chunk maintains source reference
- **Reduced noise**: Avoid overwhelming LLM with irrelevant data

**Alternative approaches considered:**
- ❌ Whole document: Too much noise for specific queries
- ❌ Sentence-level: Loses context and relationships
- ✅ Semantic sections: Balances granularity and context

---

#### 3. **Embedding Generation (`create_embeddings()`)** 
```python
def create_embeddings(self) -> None
```
**What it does:**
1. Converts each chunk into a 384-dimensional vector using `all-MiniLM-L6-v2`
2. Stores embeddings in ChromaDB with metadata
3. Creates searchable index for similarity queries

**Model Choice: `all-MiniLM-L6-v2`**
- **Speed**: 14,000+ sentences/sec on CPU
- **Quality**: 0.83 cosine similarity on semantic benchmarks
- **Size**: Only 80MB, runs locally without API costs
- **Alternative**: OpenAI embeddings (better quality, higher cost)

**How embeddings work:**
```
Text: "Test Spanish onboarding flow"
        ↓
Sentence Transformer
        ↓
Vector: [0.23, -0.45, 0.67, ..., 0.12]  (384 dimensions)
```

Similar texts have vectors close in geometric space (high cosine similarity).

---

#### 4. **Vector Storage (ChromaDB)**
```python
self.collection.add(
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
    ids=ids
)
```

**What it does:**
- In-memory vector database
- Enables fast similarity search using HNSW algorithm
- Stores document text + metadata alongside vectors

**Why ChromaDB:**
- No external server required (unlike Pinecone)
- Fast approximate nearest neighbor search
- Python-native, easy setup

---

#### 5. **Retrieval Logic (`retrieve()`)** 
```python
def retrieve(self, query: str) -> List[Dict]
```

**Step-by-step process:**

**Step 1: Query Embedding**
```python
query_embedding = self.embedding_model.encode([query])[0]
```
Converts user question into same 384-dimensional space

**Step 2: Similarity Search**
```python
results = self.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=self.top_k * 2
)
```
Finds closest vectors using cosine distance

**Step 3: Deduplication & Aggregation**
```python
# Problem: Multiple chunks per test case
# Solution: Aggregate scores by filename
for filename, scores in file_scores.items():
    avg_score = np.mean(sorted(scores, reverse=True)[:3])
```

**Why aggregate?**
- One test case might have 4 chunks
- We want to return unique test cases, not chunks
- Average of top-3 chunks gives robust relevance score

**Step 4: Ranking**
```python
aggregated_results.sort(key=lambda x: x['score'], reverse=True)
return aggregated_results[:self.top_k]
```
Returns top-5 test cases by relevance

---

#### 6. **Answer Generation (`generate_answer()`)** 
```python
def generate_answer(self, query: str, retrieved_docs: List[Dict]) -> str
```

**What it does:**

**Step 1: Build Context**
```python
context = f"""
--- Test Case 1: TC25977.json (Relevance: 0.89) ---
Title: Spanish Onboarding Flow
Summary: Validates Spanish language selection
Steps:
  1. Navigate to signup page
  2. Select "Español" from language dropdown
  ...
"""
```

**Step 2: Create Prompt**
```python
prompt = f"""
User Question: {query}
Relevant Test Cases: {full_context}

Instructions:
1. Provide clear answer
2. Cite sources as [Source: TC25977.json]
3. Be concise
"""
```

**Step 3: LLM Generation**
```python
response = self.openai_client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    temperature=0.3  # Low for consistent, factual responses
)
```

**Why GPT-4:**
- Superior reasoning and citation accuracy
- Better instruction following
- Handles complex queries with multiple test cases
- Alternative: GPT-3.5-turbo (faster, cheaper, less accurate)

---

## 🔍 How It Works: Complete Flow

### Example Query: *"What tests check Spanish onboarding?"*

```
1. USER INPUT
   └─> "What tests check Spanish onboarding?"

2. QUERY EMBEDDING
   └─> [0.34, -0.12, 0.78, ..., 0.45]  (384-dim vector)

3. VECTOR SEARCH in ChromaDB
   └─> Cosine similarity with all 100+ test case chunks
   └─> Find closest matches

4. RESULTS (before deduplication)
   • TC25977.json_overview_0: similarity 0.92
   • TC25977.json_steps_1: similarity 0.88
   • TC25977.json_metadata_3: similarity 0.85
   • TC25980.json_overview_0: similarity 0.81
   ...

5. DEDUPLICATION & AGGREGATION
   TC25977.json: avg([0.92, 0.88, 0.85]) = 0.88
   TC25980.json: avg([0.81, 0.76, 0.74]) = 0.77
   └─> Top 5 test cases selected

6. CONTEXT BUILDING
   └─> Combine full test case content for top 5

7. LLM GENERATION
   └─> GPT-4 receives: query + context
   └─> Generates: "Two test cases validate Spanish onboarding..."
   └─> Includes: [Source: TC25977.json] citations

8. RESPONSE
   └─> Answer + Retrieved test cases + Relevance scores
```

---

## 📥 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Step-by-Step Setup

```bash
# 1. Clone repository
git clone <your-repo-url>
cd rag-qa-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set OpenAI API key
export OPENAI_API_KEY='sk-...'  # On Windows: set OPENAI_API_KEY=sk-...

# 5. Verify test_cases/ directory exists
ls test_cases/  # Should show TC*.json files
```

### Dependencies Explained

```txt
openai>=1.0.0              # GPT-4 API client
sentence-transformers      # Local embedding model
chromadb                   # Vector database
numpy                      # Numerical operations
scikit-learn              # Cosine similarity, metrics
```

---

## 🚀 Usage

### Interactive Chat Interface

```bash
python app.py
```

**What happens:**
1. Loads all test cases from `test_cases/`
2. Generates embeddings (takes ~30 seconds for 100 test cases)
3. Starts interactive prompt

**Example session:**
```
🤔 Your question: What tests verify account deactivation?

🔍 Searching and generating answer...

📝 Answer:
There are three test cases that verify account deactivation:
1. [Source: TC9080.json] - Tests the complete deactivation flow
2. [Source: TC11076.json] - Verifies admin-initiated deactivation
3. [Source: TC25884.json] - Checks post-deactivation data retention

📚 Retrieved 5 test cases:
  • TC9080.json (score: 0.876) - Account Deactivation Flow
  • TC11076.json (score: 0.834) - Admin Account Management
  ...
```

### Programmatic Usage

```python
from app import TestCaseRAG

# Initialize
rag = TestCaseRAG(
    test_cases_dir="test_cases",
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4",
    top_k=5
)
rag.initialize()

# Query
result = rag.answer_question("What tests check Spanish onboarding?")
print(result['answer'])
print(result['retrieved_docs'])
```

---

## 📊 Evaluation

### Running Evaluation Suite
# Take home project specification 

The core task in this take home project is building a Retrieval Augmented Generation (RAG) system. 
We want to see your approach to solving a real world data retrieval problem using AI, without unnecessary complexity.

### The problem: *Navigating a sea of test cases*

Your biggest challenge is managing hundreds of manual test cases stored in JSON files. The entire test suite is stored in a directory containing individual JSON files (e.g., `TC25977.json`, `TC25975.json`), where each file represents a single test case.
When a product changes or a new feature is developed, finding specific test cases like "all tests related to Spanish onboarding" or "what happens when a user's account is deactivated" is a manual and time consuming process.

Simple keyword searches often fail because they lack contextual understanding. For example, a search for "account creation" might miss relevant test cases titled "onboarding flow".

### Your goal: *Build an AI powered Q&A system*

Your mission is to build a chat based assistant that allows a QA engineer to ask questions in natural language about our test suite and receive relevant, accurate answers. You will build a RAG system that uses the provided directory of JSON files as its knowledge base.

The final product should be a command line (or simple web UI) application where a user can ask a question and get back the most relevant test case(s) along with a helpful, AI generated summary.

### What you'll be working with (provided assets)

You will be provided with a package containing:

1.  `test_cases/`: A directory containing multiple `.json` files (e.g., `TC25977.json`, `TC25975.json`). Each file represents one test case and is your primary knowledge base.
2.  `sample_questions.md`: A list of strategically chosen questions (including positive, negative, and edge cases) that we will use to evaluate your solution. You should use these to test your system as you build it.

### Core requirements (deliverables)

Your solution must include the following components:

1.  **Data ingestion & preprocessing:**
    - Load and parse all `.json` files from the `test_cases/` directory.
    - Strategically decide how to chunk the documents for effective retrieval. Will you treat each file's entire content as a single document? Or will you break down the JSON into smaller, more semantic chunks (e.g., title, steps, expected results)? Explain your choice.

2.  **Embedding generation:**
    - Use a pre trained sentence transformer or an API based model (like OpenAI's) to generate vector embeddings for your processed data.

3.  **Vector store:**
    - Store the generated embeddings in a vector database or an in memory index.

4.  **Retrieval logic:**
    - Given a user's query, your system should generate an embedding for the query and use it to find the most relevant document from your vector store.

5.  **Augmented generation:**
    - Take the top k retrieved document and the original user query.
    - Pass this context to a Large Language Model (LLM).
    - The LLM should generate a clear, conversational answer that directly addresses the user's question, citing the source filename (e.g., `[Source: TC25977.json]`).

6.  **Chat interface:**
    - A simple, interactive command line interface (CLI) is the minimum requirement.
    - The interface should allow a user to ask multiple questions in a session.

7.  **Documentation (`README.md`):**
    - A clear `README.md` explaining:
        - **Setup:** How to install dependencies and configure the environment (e.g., setting an API key).
        - **Usage:** How to run your chat interface.
        - **Design Choices:** A brief explanation of your architectural decisions (e.g., "Why I chose this embedding model," "My chunking strategy and why," etc.).

### Technical stack

- Language: **Python** or **JavaScript/TypeScript** preferred (use another if you wish)
- **Core libraries:** You are free to use any libraries you see fit. We recommend considering popular frameworks built for this purpose, for example but not limited to:
    - `LangChain` or `LlamaIndex`
    - `OpenAI` API client
    - `Sentence transformers` (for local embeddings)
    - A vector store like `ChromaDB` or `Pinecone` (a simple in memory solution is acceptable)
    - Note: These are just recommendations, you are free to use any libraries you see fit.
- **Environment:** Please include a `requirements.txt` file

### Evaluation criteria

We will evaluate your submission based on the following criteria:

1.  **Correctness & relevance:** Does your system consistently retrieve the correct test cases for the queries in `sample_questions.md`? How well does it handle the negative and edge cases?
2.  **Code quality:** Is your code clean, well structured, modular, and easy to read?
3.  **Design choices & reasoning:** Your `README.md` should clearly justify your technical decisions. Why did you choose a particular embedding model, chunking strategy, or retrieval method?
4.  **Simplicity and focus:** Did you build a straightforward solution that effectively solves the core problem? We value simplicity over unnecessary complexity.
5.  **Documentation:** Is your `README.md` clear, concise, and sufficient for another engineer to run and understand your project?

### Submission guidelines

- Please create a new Git repository (on GitHub, GitLab, etc.) for your project.
- Once you are finished, please send us the link to your repository.

### Bonus points (optional)

If you have extra time and want to impress us, consider adding one of the following:

- **Web interface:** A simple web UI (e.g., using Streamlit or Flask) instead of a CLI.
- **Dockerization:** Provide a `Dockerfile` for easy setup and execution.

---   

Good luck! We're looking forward to seeing your solution.

"""
RAG-based QA Assistant for Test Case Management FINAL INTEGRATED VERSION - Complete with chunking, retrieval, and top-K keyword boosts 
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import spacy
import chromadb
from chromadb.config import Settings
import logging
from transformers import pipeline
import torch
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load small English model once
nlp = spacy.load("en_core_web_sm")

class TestCaseRAG:
    """RAG system for test case retrieval and question answering"""

    def __init__(
        self,
        test_cases_dir: str = "test_cases",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-base",
        top_k: int = 5,
        use_llm: bool = True
    ):
        self.test_cases_dir = Path(test_cases_dir)
        self.top_k = top_k
        self.llm_model_name = llm_model
        self.use_llm = use_llm

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        logger.info("Initializing ChromaDB")
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        self.collection = self.chroma_client.create_collection(
            name="test_cases",
            metadata={"hnsw:space": "cosine"}
        )
        # ALWAYS load LLM for summaries (even in structured mode)
        logger.info(f"Loading LLM for AI summaries: {llm_model}")
        device = 0 if torch.cuda.is_available() else -1

        # Initialize LLM pipeline if requested
        self.llm_pipeline = None
        if use_llm:
            logger.info(f"Loading LLM: {llm_model}")
            logger.info("First run may take 2-3 minutes (downloading model)...")
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model=llm_model,
                device=device,
                max_length=512,
                do_sample=False,
                temperature=0.3
            )
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            self.llm_model_obj = AutoModelForSeq2SeqLM.from_pretrained(llm_model)

        # Store test cases
        self.test_cases = {}

    def load_test_cases(self) -> None:
        logger.info(f"Loading test cases from {self.test_cases_dir}")
        if not self.test_cases_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.test_cases_dir}")

        json_files = list(self.test_cases_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} test case files")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    test_case = json.load(f)
                    self.test_cases[json_file.name] = test_case
            except Exception as e:
                logger.error(f"Error loading {json_file.name}: {e}")

    # ===================================================
    # SUMMARIZATION MODULE
    # ===================================================
    def summarize_test_case(self, test_case_content: dict) -> str:
        if not self.use_llm or not self.llm_pipeline:
            return "(⚠️ Summarization not available: LLM not initialized)"
        
        try:
            readable_text = ""
            if test_case_content.get("title"):
                readable_text += f"Title: {test_case_content['title']}\n"
            if test_case_content.get("custom_preconds"):
                readable_text += f"Preconditions: {test_case_content['custom_preconds']}\n"
            steps = test_case_content.get("custom_steps_separated", [])
            if steps:
                readable_text += "Steps:\n"
                for i, step in enumerate(steps, 1):
                    if isinstance(step, dict):
                        readable_text += f"  {i}. {step.get('content', '')} -> Expected: {step.get('expected', '')}\n"
            
            prompt = (
                "Summarize the following QA test case in 3-4 concise lines. "
                "Focus on what it tests, key steps, and the expected outcome.\n\n"
                f"{readable_text}"
            )
            summary = self.llm_pipeline(prompt, max_length=150, do_sample=False)[0]['generated_text']
            return summary.strip()
        except Exception as e:
            return f"(⚠️ Could not generate summary: {e})"
        

    def chunk_test_case(self, test_case: Dict, filename: str) -> List[Dict]:
        """
        Chunk a single test case into multiple documents for retrieval
        """
        chunks = []

        # Chunk 1: Overview
        overview_parts = []
        if test_case.get('title'):
            title = test_case['title']
            overview_parts.append(f"Title: {title}")
            title_lower = title.lower()
            if 'without' in title_lower or "don't" in title_lower or 'no ' in title_lower:
                overview_parts.append("Note: This test covers scenarios WITHOUT certain features")
            if 'spanish' in title_lower or 'español' in title_lower:
                overview_parts.append("Language: Spanish")
            if 'english' in title_lower:
                overview_parts.append("Language: English")
        if test_case.get('id'):
            overview_parts.append(f"Test Case ID: TC{test_case['id']}")
        if test_case.get('section_name'):
            overview_parts.append(f"Section: {test_case['section_name']}")
        if test_case.get('section_full_path'):
            overview_parts.append(f"Full Path: {test_case['section_full_path']}")
        if test_case.get('section_description'):
            overview_parts.append(f"Description: {test_case['section_description']}")
        if overview_parts:
            chunks.append({
                'text': '\n'.join(overview_parts),
                'type': 'overview',
                'filename': filename
            })

        # Chunk 2: Preconditions
        if test_case.get('custom_preconds'):
            preconds = test_case['custom_preconds']
            if preconds and preconds.strip():
                precond_text = f"Preconditions: {preconds}\n"
                preconds_lower = preconds.lower()
                if 'spanish' in preconds_lower or 'español' in preconds_lower:
                    precond_text += "Context: Spanish language testing\n"
                if 'offline' in preconds_lower:
                    precond_text += "Context: Offline mode testing\n"
                if 'coach' in preconds_lower or 'call' in preconds_lower:
                    precond_text += "Context: Coach call functionality\n"
                chunks.append({
                    'text': precond_text,
                    'type': 'preconditions',
                    'filename': filename
                })

        # Chunk 3: Steps and Expected
        steps_field = test_case.get('custom_steps_separated', [])
        if steps_field:
            steps_text = "Test Steps and Expected Results:\n"
            has_validation = False
            has_errors = False
            languages = set()
            for i, step in enumerate(steps_field, 1):
                if isinstance(step, dict):
                    step_content = step.get('content', '')
                    step_expected = step.get('expected', '')
                    if step_content:
                        steps_text += f"Step {i}: {step_content}\n"
                    if step_expected:
                        steps_text += f"Expected Result {i}: {step_expected}\n"
                        exp_lower = step_expected.lower()
                        if 'error' in exp_lower or 'invalid' in exp_lower:
                            has_errors = True
                        if 'validate' in exp_lower or 'validation' in exp_lower:
                            has_validation = True
                        if 'spanish' in exp_lower or 'español' in exp_lower:
                            languages.add('Spanish')
                        if 'english' in exp_lower:
                            languages.add('English')
                    if step_content:
                        content_lower = step_content.lower()
                        if 'spanish' in content_lower or 'español' in content_lower:
                            languages.add('Spanish')
                        if 'offline' in content_lower or 'internet' in content_lower or 'connection' in content_lower:
                            steps_text += "  [Related to: offline mode, internet connectivity]\n"
                        if 'coach' in content_lower or 'call' in content_lower or 'screening' in content_lower:
                            steps_text += "  [Related to: coach call, screening call]\n"
            if has_errors:
                steps_text += "\n[This test includes error message validation]\n"
            if has_validation:
                steps_text += "[This test validates user input]\n"
            if languages:
                steps_text += f"[Languages tested: {', '.join(languages)}]\n"
            chunks.append({
                'text': steps_text,
                'type': 'steps',
                'filename': filename
            })

        # Chunk 4: Keywords
        keyword_parts = []
        if test_case.get('title'):
            title_lower = test_case['title'].lower()
            if any(word in title_lower for word in ['onboarding', 'sign up', 'signup', 'create account', 'registration']):
                keyword_parts.append("Concept: User account creation, onboarding, signup")
            if 'resume' in title_lower:
                if 'without' in title_lower or "don't" in title_lower:
                    keyword_parts.append("Scenario: Account creation WITHOUT resume")
                else:
                    keyword_parts.append("Scenario: Account creation WITH resume")
            if any(word in title_lower for word in ['offline', 'connection', 'internet', 'network']):
                keyword_parts.append("Concept: Offline mode, internet connectivity, network issues")
            if any(word in title_lower for word in ['validate', 'validation', 'error', 'invalid']):
                keyword_parts.append("Concept: Input validation, error handling")
            if 'spanish' in title_lower:
                keyword_parts.append("Feature: Spanish language support, localization")
            if 'coach' in title_lower or 'call' in title_lower:
                keyword_parts.append("Feature: Coach call, screening call, AI interview")
        if test_case.get('section_full_path'):
            keyword_parts.append(f"Test Category: {test_case['section_full_path']}")
        if keyword_parts:
            chunks.append({
                'text': '\n'.join(keyword_parts),
                'type': 'keywords',
                'filename': filename
            })

        # Chunk 5: Metadata
        metadata_parts = []
        if test_case.get('section_name'):
            metadata_parts.append(f"Section: {test_case['section_name']}")
        if test_case.get('title'):
            metadata_parts.append(f"Test: {test_case['title']}")
        if metadata_parts:
            chunks.append({
                'text': '\n'.join(metadata_parts),
                'type': 'metadata',
                'filename': filename
            })

        return chunks

    def create_embeddings(self) -> None:
        logger.info("Creating embeddings for test cases")
        all_chunks = []
        for filename, test_case in self.test_cases.items():
            chunks = self.chunk_test_case(test_case, filename)
            all_chunks.extend(chunks)
        logger.info(f"Created {len(all_chunks)} chunks from {len(self.test_cases)} test cases")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        ids = [f"{chunk['filename']}_{chunk['type']}_{i}" for i, chunk in enumerate(all_chunks)]
        metadatas = [{'filename': chunk['filename'], 'type': chunk['type']} for chunk in all_chunks]
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("Embeddings created and stored successfully")
    
    
    def retrieve(self, query: str, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        SMART: Adaptive retrieval with dynamic keyword boosts, query expansion, and adaptive filtering.
        """

        query_lower = query.lower()

        # --------- DYNAMIC NEGATIVE QUERY DETECTION ------------
        doc = nlp(query_lower)
        is_negative_query = any(tok.lemma_ in ["any", "exist", "have", "show"] for tok in doc if tok.pos_ in {"VERB", "AUX", "PRON"}) \
                            and any(tok.dep_ in {"nsubj", "dobj"} for tok in doc)

        # --------- DYNAMIC KEYWORD EXTRACTION ------------
        keywords = set([token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"}])
        keywords.update([chunk.text.lower() for chunk in doc.noun_chunks])

        # --------- QUERY EXPANSION (dynamic) ------------
        expanded_queries = [query]
        if "don't have" in query_lower or "do not have" in query_lower:
            expanded_queries.append(query_lower.replace("don't have", "without").replace("do not have", "without"))
        if any(term in query_lower for term in ["lose internet", "lose connection", "no internet", "no connection"]):
            expanded_queries.append(query_lower + " offline mode network connectivity")
        if "invalid" in query_lower or "error" in query_lower:
            expanded_queries.append(query_lower + " validation error message")
        if "create account" in query_lower or "creating account" in query_lower:
            expanded_queries.append(query_lower + " sign up onboarding registration")

        # --------- EMBEDDING QUERY ------------
        query_embeddings = self.embedding_model.encode(expanded_queries)
        query_embedding = np.mean(query_embeddings, axis=0)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k * 3
        )

        # --------- DYNAMIC KEYWORD BOOST PER FILE ------------
        file_scores = {}
        file_chunks = {}
        file_keyword_boosts = {}

        for doc_text, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            filename = metadata['filename']
            similarity = 1 - distance
            if similarity < similarity_threshold:
                continue

            if filename not in file_scores:
                file_scores[filename] = []
                file_chunks[filename] = []
                file_keyword_boosts[filename] = 0.0

            file_scores[filename].append(similarity)
            file_chunks[filename].append({'text': doc_text, 'type': metadata['type'], 'similarity': similarity})

            doc_lower = doc_text.lower()
            boost = sum(0.05 for kw in keywords if kw in doc_lower)  # dynamic boost only when query keyword matches
            file_keyword_boosts[filename] = max(file_keyword_boosts[filename], boost)

        # --------- AGGREGATE RESULTS PER FILE ------------
        aggregated_results = []
        for filename, scores in file_scores.items():
            base_score = np.mean(sorted(scores, reverse=True)[:3])
            boosted_score = base_score + file_keyword_boosts[filename]
            aggregated_results.append({
                'filename': filename,
                'score': boosted_score,
                'base_score': base_score,
                'boost': file_keyword_boosts[filename],
                'chunks': file_chunks[filename],
                'test_case': self.test_cases[filename]
            })

        aggregated_results.sort(key=lambda x: x['score'], reverse=True)

        # --------- SMART ADAPTIVE FILTERING ------------
        if not aggregated_results:
            return []

        top_score = aggregated_results[0]['score']

        if is_negative_query:
            min_threshold = 0.55
            relative_factor = 0.75
        else:
            min_threshold = 0.40
            relative_factor = 0.65

        if top_score < min_threshold:
            return []

        filtered_results = []
        for result in aggregated_results:
            relative_threshold = top_score * relative_factor
            absolute_min = 0.45 if not is_negative_query else 0.55
            if result['score'] >= max(relative_threshold, absolute_min):
                filtered_results.append(result)

        return filtered_results[:self.top_k]

    
    def generate_structured_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        FIXED: Prevents token overflow in AI summary
        """
        if not retrieved_docs:
            return (
                "I could not find any test cases related to this query in the provided test suite.\n\n"
                "This may mean:\n"
                "• The feature is not currently tested\n"
                "• The query uses different terminology than the test cases\n"
                "• Try rephrasing your question with different keywords"
            )

        # Part 1: Structured test case details
        answer = f"Found {len(retrieved_docs)} relevant test case(s):\n\n"
        
        for i, doc in enumerate(retrieved_docs, 1):
            test_case = doc['test_case']
            filename = doc['filename']
            score = doc['score']
            boost_info = f" [+{doc['boost']:.2f} keyword boost]" if 'boost' in doc and doc['boost'] > 0 else ""
            
            answer += f"{'='*60}\n"
            answer += f"{i}. [{filename}] (Relevance: {score:.2f}{boost_info})\n"
            answer += f"{'='*60}\n"
            
            if test_case.get('title'):
                answer += f"📋 Title: {test_case['title']}\n"
            
            if test_case.get('section_full_path'):
                answer += f"📁 Path: {test_case['section_full_path']}\n"
            
            if test_case.get('custom_preconds'):
                preconds = test_case['custom_preconds']
                if preconds and preconds.strip():
                    answer += f"⚙️  Preconditions: {preconds[:150]}{'...' if len(preconds) > 150 else ''}\n"
            
            steps = test_case.get('custom_steps_separated', [])
            if steps:
                answer += f"📝 Steps: {len(steps)} steps\n"
                # Show first 2 steps
                for j, step in enumerate(steps[:2], 1):
                    if isinstance(step, dict):
                        content = step.get('content', '')
                        if content:
                            preview = content[:100] + '...' if len(content) > 100 else content
                            answer += f"   {j}. {preview}\n"
            
            answer += "\n"
        
        # Part 2: AI-Generated Summary (FIXED!)
        if self.use_llm and self.llm_pipeline:
            answer += f"{'='*60}\n"
            answer += "🤖 AI-Generated Summary:\n"
            answer += f"{'='*60}\n"
            
            try:
                # Build MINIMAL context (prevent token overflow)
                summary_items = []
                for doc in retrieved_docs[:3]:  # Top 3 only
                    tc = doc['test_case']
                    title = tc.get('title', 'Untitled')[:80]  # Truncate long titles
                    summary_items.append(f"- {title} [{doc['filename']}]")
                
                context_text = '\n'.join(summary_items)
                
                # VERY SHORT prompt (under 100 tokens)
                summary_prompt = f"""Answer this question using these test cases:
    Q: {query[:100]}

    Tests:
    {context_text}

    Answer in 2 sentences:"""
                
                # Generate with strict limits
                summary_response = self.llm_pipeline(
                    summary_prompt,
                    max_length=100,      # Short output
                    min_length=20,
                    do_sample=False,
                    num_beams=1,         # Faster, prevents repetition
                    early_stopping=True
                )
                
                summary = summary_response[0]['generated_text'].strip()
                
                # Validate summary quality
                if len(summary) < 10 or summary.count('.') == 0:
                    raise ValueError("Summary too short or malformed")
                
                answer += f"{summary}\n"
                
            except Exception as e:
                logger.warning(f"AI summary failed: {e}")
                # Fallback to simple summary
                answer += f"The system found {len(retrieved_docs)} relevant test cases. "
                answer += f"Primary match: {retrieved_docs[0]['test_case'].get('title', 'See details above')}. "
                answer += "See structured details above for complete information.\n"
        else:
            # No LLM - provide simple summary
            answer += f"{'='*60}\n"
            answer += "📋 Summary:\n"
            answer += f"{'='*60}\n"
            answer += f"Found {len(retrieved_docs)} test case(s) matching your query. "
            answer += f"Primary match: {retrieved_docs[0]['test_case'].get('title', 'See details above')}\n"
        
        return answer
    
    def generate_llm_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        from nltk.stem import WordNetLemmatizer
        import re
        import logging
        lemmatizer = WordNetLemmatizer()
        logger = logging.getLogger(__name__)

        """
        Generate LLM-based answer using retrieved documents.
        Dynamically tags scenarios like offline mode or validation using lemmatization.
        Truncates context safely to avoid token overflow.
        """
        if not retrieved_docs:
            return self.generate_structured_answer(query, retrieved_docs)

        # Define scenario keywords
        keywords_to_tag = {
            "Offline scenario": ["disable internet", "offline", "no network", "disconnect", "lost connection"],
            "Validation scenario": ["invalid", "error", "required", "must enter"]
        }

        # Map tags to descriptive phrases
        tag_to_phrase = {
            "Offline scenario": "This action may occur when the app is offline or the internet is unavailable.",
            "Validation scenario": "This action triggers a validation error that must be corrected."
        }

        context_parts = []
        total_tokens = 0
        max_context_tokens = 400  # prevent overflow

        for i, doc in enumerate(retrieved_docs[:3], 1):
            test_case = doc["test_case"]
            filename = doc["filename"]

            context = f"Test Case {i} [{filename}]:\n"
            context += f"Title: {test_case.get('title', 'N/A')}\n"

            if test_case.get("section_full_path"):
                context += f"Path: {test_case['section_full_path']}\n"

            if test_case.get("custom_preconds"):
                preconds = test_case["custom_preconds"]
                context += f"Prerequisites: {preconds[:150]}{'...' if len(preconds) > 150 else ''}\n"

            steps = test_case.get("custom_steps_separated", [])
            if steps:
                context += f"Steps: {min(len(steps),5)} steps (truncated)\n"
                for j, step in enumerate(steps[:5], 1):
                    if isinstance(step, dict):
                        content = step.get("content", "")
                        expected = step.get("expected", "")
                        
                        # Lemmatize words for scenario detection
                        step_words = (content + " " + expected).lower().split()
                        step_lemmas = [lemmatizer.lemmatize(w) for w in step_words]
                        step_text_for_matching = " ".join(step_lemmas)

                        # Detect scenarios
                        tags = [tag for tag, kws in keywords_to_tag.items()
                                if any(kw in step_text_for_matching for kw in kws)]
                        # Produce descriptive phrases
                        tag_phrase = " ".join([tag_to_phrase[tag] for tag in tags]) + " " if tags else ""

                        content_preview = content[:50] + ("..." if len(content) > 50 else "")
                        expected_preview = expected[:75] + ("..." if len(expected) > 75 else "")
                        context += f"  Step {j}: {tag_phrase}{content_preview} -> Expected: {expected_preview}\n"

            if test_case.get("description"):
                desc = test_case["description"]
                context += f"Description: {desc[:150]}{'...' if len(desc) > 150 else ''}\n"

            # Stop adding more context if token budget exceeded
            token_estimate = len(context.split()) // 1.3
            if total_tokens + token_estimate > max_context_tokens:
                break
            total_tokens += token_estimate
            context_parts.append(context)

        full_context = "\n".join(context_parts)

        prompt = f"""You are a QA assistant. Answer based ONLY on the test cases below.

    RULES:
    1. Only use information from the provided test cases.
    2. Steps may have descriptive phrases like: 
    - Offline scenario → app is offline or internet unavailable
    - Validation scenario → validation errors
    3. Always cite source files in square brackets, e.g. [Source: filename].
    4. If test cases don't answer the question, say so clearly.
    5. Never invent new test case names or details.

    Question: {query}

    Test Cases:
    {full_context}

    Answer:"""

        try:
            response = self.llm_pipeline(
                prompt,
                max_length=300,
                min_length=30,
                do_sample=False,
                num_beams=2,
                early_stopping=True
            )
            answer = response[0]["generated_text"].strip()

            # --- Post-cleaning for repetition ---
            lines = answer.splitlines()
            cleaned = []
            seen = set()
            for line in lines:
                if line.strip().lower() not in seen:
                    cleaned.append(line)
                    seen.add(line.strip().lower())
            answer = "\n".join(cleaned)

            # If the model loops (same phrase repeated)
            if answer.lower().count(query.lower()) > 2 or len(set(answer.split())) < 20:
                logger.warning("Detected repetition or meaningless response. Falling back.")
                return self.generate_structured_answer(query, retrieved_docs)

            # Verify citations
            import re
            cited_files = set(re.findall(r"\bTC\d+\.json\b", answer))
            valid_files = {doc["filename"] for doc in retrieved_docs}
            invalid_citations = cited_files - valid_files

            if invalid_citations:
                logger.warning(f"Hallucination detected in LLM answer: {invalid_citations}")
                return self.generate_structured_answer(query, retrieved_docs)

            return answer

        except Exception as e:
            logger.error(f"Error generating LLM answer: {e}")
            return self.generate_structured_answer(query, retrieved_docs)


    def answer_question(self, query: str) -> Dict:
        logger.info(f"Processing query: {query}")
        retrieved_docs = self.retrieve(query)
        if self.use_llm and self.llm_pipeline:
            answer = self.generate_llm_answer(query, retrieved_docs)
        else:
            answer = self.generate_structured_answer(query, retrieved_docs)
        return {
            'query': query,
            'answer': answer,
            'retrieved_docs': [{'filename': doc['filename'], 'score': doc['score'], 'title': doc['test_case'].get('title', 'N/A')} for doc in retrieved_docs],
            'num_retrieved': len(retrieved_docs)
        }

    def initialize(self) -> None:
        self.load_test_cases()
        self.create_embeddings()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RAG QA Assistant')
    parser.add_argument('--mode', choices=['structured', 'llm'], default='structured', 
                       help='Output mode: structured (safe) or llm (natural language)')
    parser.add_argument('--model', default='google/flan-t5-base', 
                       help='LLM model to use')
    args = parser.parse_args()

    print("="*70)
    print("QA Test Case Assistant - RAG System")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'structured':
        print("✓ Structured details + AI summary")
    else:
        print("✓ Natural language generation")
    print("="*70)

    print("\n🔧 Initializing RAG system...")
    try:
        # ALWAYS load LLM (needed for summaries in structured mode)
        rag = TestCaseRAG(
            test_cases_dir="test_cases",
            embedding_model="all-MiniLM-L6-v2",
            llm_model=args.model,  # Always pass model
            top_k=5,
            use_llm=True  # Always True (needed for summaries)
        )
        rag.initialize()
        print("✅ System initialized successfully!")
        print(f"📊 Loaded {len(rag.test_cases)} test cases")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*70)
    print("💬 Chat Interface")
    print("="*70)
    print("Ask questions about test cases (or 'quit' to exit)")
    print("-"*70)

    while True:
        try:
            query = input("\n🤔 Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            if not query:
                continue
            
            print("\n🔍 Searching...")
            result = rag.answer_question(query)
            
            print("\n" + "="*70)
            print("📝 Answer:")
            print("="*70)
            
            # Just print the answer - it already includes everything!
            print(result['answer'])
            
            print("\n" + "-"*70)
            print(f"📚 Retrieved {result['num_retrieved']} test cases:")
            for doc in result['retrieved_docs']:
                print(f"  • {doc['filename']} (score: {doc['score']:.3f}) - {doc['title']}")
            print("-"*70)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error processing query")


if __name__ == "__main__":
    main()

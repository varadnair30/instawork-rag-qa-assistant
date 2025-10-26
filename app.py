"""
RAG-based QA Assistant for Test Case Management
FINAL INTEGRATED VERSION - Complete with chunking, retrieval, and top-K keyword boosts
"""

import sys
import json
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
import chromadb
from chromadb.config import Settings
import logging
from transformers import pipeline
import torch

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
        SMART: Adaptive thresholds based on query characteristics
        """
        query_lower = query.lower()

        # Detect query type
        is_negative_query = any(word in query_lower for word in [
            "any test", "are there", "do we have", "do we test", 
            "is there", "show me test"
        ]) and not any(word in query_lower for word in [
            "spanish", "resume", "onboarding", "offline", "email", 
            "validation", "error", "internet", "coach"
        ])
        
        # Extract keywords
        doc = nlp(query_lower)
        keywords = set([token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"}])
        keywords.update([chunk.text.lower() for chunk in doc.noun_chunks])

        # Query expansion
        expanded_queries = [query]
        if "don't have" in query_lower or "do not have" in query_lower:
            expanded_queries.append(query_lower.replace("don't have", "without").replace("do not have", "without"))
        if any(term in query_lower for term in ["lose internet", "lose connection", "no internet", "no connection"]):
            expanded_queries.append(query_lower + " offline mode network connectivity")
        if "invalid" in query_lower or "error" in query_lower:
            expanded_queries.append(query_lower + " validation error message")
        if "create account" in query_lower or "creating account" in query_lower:
            expanded_queries.append(query_lower + " sign up onboarding registration")

        # Compute embeddings
        query_embeddings = self.embedding_model.encode(expanded_queries)
        query_embedding = np.mean(query_embeddings, axis=0)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=self.top_k * 3
        )

        # Boost dictionary
        boost_dict = {
            "without": 0.15,
            "don't have": 0.15,
            "invalid": 0.10,
            "error": 0.10,
            "offline": 0.20,
            "internet": 0.20,
            "connection": 0.20,
            "spanish": 0.10,
            "coach": 0.10,
            "call": 0.10,
            "resume": 0.10,
            "validation": 0.10,
            "signup": 0.10,
            "registration": 0.10,
            "qr": 0.15,
            "code": 0.10
        }

        # Process chunks
        file_scores = {}
        file_chunks = {}
        file_keyword_boosts = {}

        for i, (doc_text, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
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
            boost = 0.0
            for kw in keywords:
                if kw in doc_lower:
                    boost += boost_dict.get(kw, 0.05)

            file_keyword_boosts[filename] = max(file_keyword_boosts[filename], boost)

        # Aggregate results
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

        # ============ SMART ADAPTIVE FILTERING ============
        if not aggregated_results:
            return []
        
        top_score = aggregated_results[0]['score']
        
        # Context-aware thresholds
        if is_negative_query:
            # Strict for "do we have X" queries
            min_threshold = 0.55
            relative_factor = 0.75
        else:
            # Lenient for specific feature queries
            min_threshold = 0.40
            relative_factor = 0.65
        
        # Filter by minimum threshold
        if top_score < min_threshold:
            return []
        
        # Adaptive relative filtering
        filtered_results = []
        for result in aggregated_results:
            relative_threshold = top_score * relative_factor
            absolute_min = 0.45 if not is_negative_query else 0.55
            
            if result['score'] >= max(relative_threshold, absolute_min):
                filtered_results.append(result)
        
        return filtered_results[:self.top_k]
    def generate_structured_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return (
                "I could not find any test cases related to this query in the provided test suite.\n\n"
                "This may mean:\n"
                "• The feature is not currently tested\n"
                "• The query uses different terminology than the test cases\n"
                "• Try rephrasing your question with different keywords"
            )

        answer = f"Found {len(retrieved_docs)} relevant test case(s):\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            test_case = doc['test_case']
            filename = doc['filename']
            score = doc['score']
            boost_info = f" [+{doc['boost']:.2f} keyword boost]" if 'boost' in doc and doc['boost'] > 0 else ""
            answer += f"{'='*60}\n{i}. [{filename}] (Relevance: {score:.2f}{boost_info})\n{'='*60}\n"
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
                for j, step in enumerate(steps[:2], 1):
                    if isinstance(step, dict):
                        content = step.get('content', '')
                        if content:
                            preview = content[:100] + '...' if len(content) > 100 else content
                            answer += f"   {j}. {preview}\n"
            answer += "\n"
        return answer

    def generate_llm_answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):
            test_case = doc['test_case']
            filename = doc['filename']
            context = f"Test Case {i} [{filename}]:\n"
            context += f"Title: {test_case.get('title', 'N/A')}\n"
            if test_case.get('section_full_path'):
                context += f"Path: {test_case['section_full_path']}\n"
            if test_case.get('custom_preconds'):
                context += f"Prerequisites: {test_case['custom_preconds'][:100]}\n"
            steps = test_case.get('custom_steps_separated', [])
            if steps:
                context += f"Steps: {len(steps)} test steps\n"
            context_parts.append(context)
        full_context = '\n'.join(context_parts)
        prompt = f"""You are a QA assistant. Answer based ONLY on the test cases below.

RULES:
1. Only use information from the provided test cases
2. Always cite source files [Source: filename]
3. If test cases don't answer the question, say so
4. Never invent test case names or details

Question: {query}

Test Cases:
{full_context}

Answer:"""

        try:
            response = self.llm_pipeline(
                prompt,
                max_length=400,
                min_length=30,
                do_sample=False
            )
            answer = response[0]['generated_text']
            import re
            cited_files = set(re.findall(r'TC\d+\.json', answer))
            valid_files = {doc['filename'] for doc in retrieved_docs}
            invalid_citations = cited_files - valid_files
            if invalid_citations:
                logger.warning(f"Hallucination detected: {invalid_citations}")
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
    parser.add_argument('--mode', choices=['structured', 'llm'], default='structured', help='Output mode: structured (safe) or llm (natural language)')
    parser.add_argument('--model', default='google/flan-t5-base', help='LLM model to use (only for llm mode)')
    args = parser.parse_args()

    print("="*70)
    print("QA Test Case Assistant - RAG System")
    print("="*70)
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'structured':
        print("✓ Structured output (zero hallucination risk)")
    else:
        print("✓ Natural language generation with validation")
    print("="*70)

    print("\n🔧 Initializing RAG system...")
    try:
        use_llm = (args.mode == 'llm')
        rag = TestCaseRAG(
            test_cases_dir="test_cases",
            embedding_model="all-MiniLM-L6-v2",
            llm_model=args.model if use_llm else None,
            top_k=5,
            use_llm=use_llm
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

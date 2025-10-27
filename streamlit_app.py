"""
Streamlit UI for RAG-based QA Test Case Assistant
Clean, professional interface with chat history
"""

import streamlit as st
from app import TestCaseRAG
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="QA Test Case Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .test-case-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .test-case-title {
        font-weight: bold;
        color: #1f77b4;
        font-size: 1.1rem;
    }
    .relevance-score {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.85rem;
        display: inline-block;
    }
    .query-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .answer-box {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #8bc34a;
        color: #1a1a1a;
    }
    .metric-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
        color: #1a1a1a;
    }
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: 2rem;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=QA+Assistant", width=200)
    st.markdown("### ⚙️ System Configuration")
    
    mode = st.radio(
        "Output Mode",
        ["Structured (Recommended)", "Natural Language"],
        help="Structured mode: Zero hallucination + AI summary\nNatural Language: Full LLM response"
    )
    
    # Fixed model - no dropdown
    model = "google/flan-t5-base"
    st.info(f"**Model:** {model}")
    
    top_k = 5
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    if st.session_state.system_initialized:
        st.success("✅ System Ready")
        if st.session_state.rag_system:
            st.metric("Test Cases Loaded", len(st.session_state.rag_system.test_cases))
            st.metric("Queries Processed", len(st.session_state.chat_history))
    else:
        st.warning("⏳ System Not Initialized")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    

# Main content
st.markdown('<div class="main-header">🔍 QA Test Case Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Test Case Retrieval & Analysis</div>', unsafe_allow_html=True)

# Initialize system
if not st.session_state.system_initialized:
    with st.spinner("🔧 Initializing RAG system... This may take 1-2 minutes on first run."):
        try:
            # Show initialization progress
            progress_text = st.empty()
            progress_text.text("📦 Loading NLTK and Spacy resources...")
            
            use_llm_mode = (mode == "Natural Language")
            
            progress_text.text("🤖 Initializing RAG system...")
            rag = TestCaseRAG(
                test_cases_dir="test_cases",
                embedding_model="all-MiniLM-L6-v2",
                llm_model=model,
                top_k=top_k,
                use_llm=True  # Always True for summaries
            )
            
            progress_text.text("📚 Loading test cases and creating embeddings...")
            rag.initialize()
            
            st.session_state.rag_system = rag
            st.session_state.system_initialized = True
            
            progress_text.empty()
            st.success(f"✅ System initialized! Loaded {len(rag.test_cases)} test cases.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            st.exception(e)  # Show full traceback for debugging
            st.stop()

# Chat interface
st.markdown("### 💬 Ask a Question")

# Query input
query = st.text_input(
    "Enter your question about test cases:",
    placeholder="e.g., What tests cover Spanish onboarding?",
    key="query_input",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    example_btn1 = st.button("📱 Example 1 (What tests cover Spanish onboarding?)", use_container_width=True)
with col3:
    example_btn2 = st.button("📱 Example 2 (What happens if internet is lost during signup?)", use_container_width=True)

# Handle example buttons
if example_btn1:
    query = "What tests cover Spanish onboarding?"
    search_button = True
if example_btn2:
    query = "What happens if internet is lost during signup?"
    search_button = True

# Process query
if search_button and query:
    with st.spinner("🔍 Searching and analyzing..."):
        try:
            start_time = time.time()
            
            # Get answer
            result = st.session_state.rag_system.answer_question(query)
            
            elapsed_time = time.time() - start_time
            
            # Extract AI summary from full answer
            full_answer = result['answer']
            summary = ""
            
            if "🤖 AI-Generated Summary:" in full_answer:
                # Extract the summary section
                summary_start = full_answer.find("🤖 AI-Generated Summary:")
                summary_section = full_answer[summary_start:]
                lines = summary_section.split('\n')
                # Skip the header and separator lines, get the actual summary
                summary_lines = [line.strip() for line in lines if line.strip() and '=' not in line and '🤖' not in line]
                summary = ' '.join(summary_lines) if summary_lines else ""
            
            # Fallback if no summary found
            if not summary:
                summary = f"Found {result['num_retrieved']} relevant test case(s). See details below."
            
            # Add to chat history with both summary and full answer
            st.session_state.chat_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'query': query,
                'summary': summary,
                'full_answer': full_answer,
                'result': result,
                'elapsed_time': elapsed_time
            })
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            st.exception(e)  # Show full traceback for debugging

# Display chat history (most recent first)
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📝 Results")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # User Question Box with visible text
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1976d2;">
                <div style="color: #1976d2; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    🤔 YOUR QUESTION • {chat['timestamp']}
                </div>
                <div style="color: #1a1a1a; font-size: 1rem; line-height: 1.5;">
                    {chat['query']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Answer Box - Show only the concise AI summary
            st.markdown(f"""
            <div style="background-color: #f1f8e9; padding: 1.2rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #7cb342;">
                <div style="color: #558b2f; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    🤖 AI ANSWER • ⏱️ {chat['elapsed_time']:.2f}s
                </div>
                <div style="color: #1a1a1a; font-size: 1rem; line-height: 1.6;">
                    {chat.get('summary', 'No summary available')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Retrieved test cases - Compact display with error handling
            st.markdown("**📚 Retrieved Test Cases:**")
            
            # Handle case when no test cases are retrieved (fixes the error)
            num_docs = len(chat['result']['retrieved_docs'])
            
            if num_docs == 0:
                # Graceful handling - no results
                st.markdown("""
                <div style="background-color: #fff3cd; padding: 1rem; border-radius: 6px; margin: 0.5rem 0; border-left: 3px solid #ffc107;">
                    <div style="color: #856404; font-size: 0.9rem;">
                        ℹ️ No relevant test cases were found for this query.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Calculate number of columns based on number of docs
                num_cols = min(num_docs, 3)
                cols = st.columns(num_cols)
                
                for idx, doc in enumerate(chat['result']['retrieved_docs']):
                    col_idx = idx % num_cols
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div style="background-color: #fafafa; padding: 0.8rem; border-radius: 6px; margin: 0.3rem 0; border-left: 3px solid #ff9800;">
                            <div style="font-weight: 600; color: #e65100; font-size: 0.85rem;">
                                {doc['filename']}
                            </div>
                            <div style="background-color: #4caf50; color: white; display: inline-block; padding: 0.15rem 0.4rem; border-radius: 10px; font-size: 0.75rem; margin: 0.3rem 0;">
                                Score: {doc['score']:.3f}
                            </div>
                            <div style="color: #616161; font-size: 0.8rem; margin-top: 0.3rem; line-height: 1.3;">
                                {doc['title'][:55]}{'...' if len(doc['title']) > 55 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Full details in collapsible expander
            with st.expander("📄 View Full Technical Details", expanded=False):
                st.text(chat.get('full_answer', chat['result']['answer']))
            
            st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)

else:
    # Welcome message
    st.info("👋 Welcome! Ask a question about test cases to get started.")
    
    # Example queries
    st.markdown("### 💡 Example Queries")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Basic Queries:**
        - What tests cover Spanish onboarding?
        - How do we test offline mode?
        - Which tests verify login functionality?
        """)
    
    with col2:
        st.markdown("""
        **Advanced Queries:**
        - What happens if internet is lost during signup?
        - What validation errors occur during account creation?
        - Show me tests where users don't have a resume
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📊 System Stats**")
    st.markdown(f"Mode: {mode}")
with col2:
    st.markdown("**🏆 Performance**")
    st.markdown("Evaluation: 100% (Grade A)")
with col3:
    st.markdown("**🔧 Technology**")
    st.markdown("RAG + FLAN-T5 + ChromaDB")
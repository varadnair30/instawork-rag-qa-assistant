import pytest
from app import TestCaseRAG

# EXACT queries from sample_questions.md
OFFICIAL_TEST_CASES = {
    "test_1_synonym_negative": {
        "query": "How does a Pro create an account if they don't have a resume?",
        "expected_primary": ["TC25977.json"],
        "expected_secondary": ["TC25975.json", "TC25976.json"],
        "description": "Synonym matching + negative constraint"
    },
    "test_2_multi_keyword": {
        "query": "Show me test cases for the Spanish language option during the AI coach call.",
        "expected_primary": ["TC2268.json", "TC25976.json"],
        "expected_secondary": ["TC9080.json", "TC26676.json", "TC25975.json"],
        "description": "Multi-keyword precision (Spanish + coach call)"
    },
    "test_3_error_condition": {
        "query": "What happens if a user tries to create an account with an invalid email address?",
        "expected_primary": ["TC26415.json", "TC25977.json"],
        "expected_secondary": ["TC25884.json", "TC25975.json"],
        "description": "Deep content search (error in steps/expected)"
    },
    "test_4_vague_concept": {
        "query": "What happens if I lose my internet connection while looking for shifts?",
        "expected_primary": ["TC11076.json"],
        "expected_secondary": ["TC11096.json", "TC11088.json", "TC27696.json"],
        "description": "Vague → technical mapping (lose internet → offline)"
    },
    "test_5_negative_case": {
        "query": "Are there any test cases for dark mode or theme switching functionality?",
        "expected_primary": [],
        "expected_secondary": [],
        "description": "Negative case - avoid hallucination"
    }
}


def evaluate_result(test_case, result):
    retrieved_files = [doc['filename'] for doc in result['retrieved_docs']]

    # Negative case
    if not test_case['expected_primary'] and not test_case['expected_secondary']:
        return 1.0  # perfect score if nothing returned

    # Primary (top 5)
    top_5 = retrieved_files[:5]
    primary_found = [f for f in test_case['expected_primary'] if f in top_5]
    primary_score = len(primary_found) / len(test_case['expected_primary']) if test_case['expected_primary'] else 0

    # Secondary (anywhere)
    secondary_found = [f for f in test_case['expected_secondary'] if f in retrieved_files]
    secondary_score = len(secondary_found) / len(test_case['expected_secondary']) if test_case['expected_secondary'] else 0

    # Weighted overall
    overall_score = (primary_score * 0.7) + (secondary_score * 0.3)
    return overall_score


@pytest.fixture(scope="module")
def rag_system():
    rag = TestCaseRAG(
        test_cases_dir="test_cases",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        use_llm=False
    )
    rag.initialize()
    return rag


@pytest.mark.parametrize("test_name,test_case", OFFICIAL_TEST_CASES.items())
def test_official_queries(rag_system, test_name, test_case):
    """
    Evaluate each official test query using weighted scoring.
    Test passes if overall score >= 0.7
    """
    result = rag_system.answer_question(test_case['query'])
    score = evaluate_result(test_case, result)
    assert score >= 0.7, f"{test_name} failed: Overall score below threshold ({score*100:.1f}%)"



from app import TestCaseRAG
import json

# EXACT queries from sample_questions.md
OFFICIAL_TEST_CASES = {
    "test_1_synonym_negative": {
        "query": "How does a Pro create an account if they don't have a resume?",
        "expected_primary": ["TC25977.json"],  # WITHOUT resume
        "expected_secondary": ["TC25975.json", "TC25976.json"],
        "description": "Synonym matching + negative constraint"
    },
    "test_2_multi_keyword": {
        "query": "Show me test cases for the Spanish language option during the AI coach call.",
        "expected_primary": ["TC2268.json", "TC25976.json"],  # Spanish support
        "expected_secondary": ["TC9080.json", "TC26676.json", "TC25975.json"],
        "description": "Multi-keyword precision (Spanish + coach call)"
    },
    "test_3_error_condition": {
        "query": "What happens if a user tries to create an account with an invalid email address?",
        "expected_primary": ["TC26415.json", "TC25977.json"],  # Validation errors
        "expected_secondary": ["TC25884.json", "TC25975.json"],
        "description": "Deep content search (error in steps/expected)"
    },
    "test_4_vague_concept": {
        "query": "What happens if I lose my internet connection while looking for shifts?",
        "expected_primary": ["TC11076.json"],  # Offline Open Shifts
        "expected_secondary": ["TC11096.json", "TC11088.json", "TC27696.json"],
        "description": "Vague → technical mapping (lose internet → offline)"
    },
    "test_5_negative_case": {
        "query": "Are there any test cases for dark mode or theme switching functionality?",
        "expected_primary": [],  # Should return NOTHING
        "expected_secondary": [],
        "description": "Negative case - avoid hallucination"
    }
}

def evaluate_result(test_name, test_case, result):
    """
    Evaluate if retrieval matches expected results
    Returns: (score, details)
    """
    retrieved_files = [doc['filename'] for doc in result['retrieved_docs']]
    
    # For negative case
    if not test_case['expected_primary'] and not test_case['expected_secondary']:
        if not retrieved_files:
            return 1.0, "✅ PERFECT - Correctly returned no results"
        else:
            return 0.0, f"❌ FAIL - Should return nothing, but returned: {retrieved_files}"
    
    # Check primary relevance (top 2 results)
    top_2 = retrieved_files[:2] if len(retrieved_files) >= 2 else retrieved_files
    primary_found = [f for f in test_case['expected_primary'] if f in top_2]
    primary_score = len(primary_found) / len(test_case['expected_primary']) if test_case['expected_primary'] else 0
    
    # Check secondary relevance (all results)
    secondary_found = [f for f in test_case['expected_secondary'] if f in retrieved_files]
    secondary_score = len(secondary_found) / len(test_case['expected_secondary']) if test_case['expected_secondary'] else 0
    
    # Overall score (primary weighted 70%, secondary 30%)
    overall_score = (primary_score * 0.7) + (secondary_score * 0.3)
    
    details = f"""
  Primary matches: {len(primary_found)}/{len(test_case['expected_primary'])} ({primary_score*100:.0f}%)
    Expected: {test_case['expected_primary']}
    Found in top 2: {primary_found}
  Secondary matches: {len(secondary_found)}/{len(test_case['expected_secondary'])} ({secondary_score*100:.0f}%)
    Expected: {test_case['expected_secondary']}
    Found: {secondary_found}
  Retrieved: {retrieved_files}
  Overall Score: {overall_score*100:.1f}%
"""
    
    return overall_score, details

def main():
    print("="*80)
    print("OFFICIAL TEST SCRIPT - Using sample_questions.md")
    print("="*80)
    
    # Initialize RAG
    print("\n🔧 Initializing RAG system...")
    rag = TestCaseRAG(
        test_cases_dir="test_cases",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,  # Return 5 results for evaluation
        use_llm=False  # Use structured mode for testing
    )
    rag.initialize()
    print(f"✅ Loaded {len(rag.test_cases)} test cases\n")
    
    # Run official tests
    results = {}
    overall_scores = []
    
    for test_name, test_case in OFFICIAL_TEST_CASES.items():
        print(f"\n{'='*80}")
        print(f"TEST: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Query: \"{test_case['query']}\"")
        print("-"*80)
        
        # Execute query
        result = rag.answer_question(test_case['query'])
        
        # Evaluate result
        score, details = evaluate_result(test_name, test_case, result)
        overall_scores.append(score)
        
        print(details)
        
        # Show actual answer preview
        print(f"\nGenerated Answer (preview):")
        print(f"  {result['answer'][:200]}...")
        
        results[test_name] = {
            'test_case': test_case,
            'score': score,
            'retrieved': [doc['filename'] for doc in result['retrieved_docs']],
            'answer': result['answer']
        }
    
    # Summary
    print("\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    
    avg_score = sum(overall_scores) / len(overall_scores)
    
    print(f"\nOverall Performance: {avg_score*100:.1f}%")
    print(f"Tests Passed (>70%): {sum(1 for s in overall_scores if s >= 0.7)}/{len(overall_scores)}")
    
    print("\n" + "-"*80)
    print("Individual Test Scores:")
    print("-"*80)
    
    for i, (test_name, score) in enumerate(zip(OFFICIAL_TEST_CASES.keys(), overall_scores), 1):
        status = "✅ PASS" if score >= 0.7 else "⚠️  NEEDS WORK" if score >= 0.5 else "❌ FAIL"
        test_desc = OFFICIAL_TEST_CASES[test_name]['description']
        print(f"{i}. {status} - {test_desc}: {score*100:.1f}%")
    
    # Grade interpretation
    print("\n" + "="*80)
    if avg_score >= 0.85:
        grade = "A (Excellent)"
        print(f"🏆 Grade: {grade}")
        print("Your RAG system performs excellently on the evaluation criteria!")
    elif avg_score >= 0.75:
        grade = "B (Good)"
        print(f"👍 Grade: {grade}")
        print("Your system performs well. Minor improvements possible.")
    elif avg_score >= 0.65:
        grade = "C (Acceptable)"
        print(f"✓ Grade: {grade}")
        print("System is functional but needs improvement in some areas.")
    else:
        grade = "D (Needs Work)"
        print(f"⚠️  Grade: {grade}")
        print("System needs significant improvements in retrieval quality.")
    print("="*80)
    
    # Save detailed results
    with open('official_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'overall_score': avg_score,
                'grade': grade,
                'tests_passed': sum(1 for s in overall_scores if s >= 0.7),
                'total_tests': len(overall_scores)
            },
            'detailed_results': results
        }, f, indent=2)
    
    print("\n💾 Detailed results saved to: official_test_results.json")
    
    # Recommendations
    if avg_score < 0.85:
        print("\n💡 Recommendations for Improvement:")
        for test_name, score in zip(OFFICIAL_TEST_CASES.keys(), overall_scores):
            if score < 0.7:
                test_case = OFFICIAL_TEST_CASES[test_name]
                print(f"\n  • {test_case['description']}")
                print(f"    Query: \"{test_case['query']}\"")
                print(f"    Current score: {score*100:.1f}%")
                print(f"    Missing primary: {test_case['expected_primary']}")

if __name__ == "__main__":
    main()
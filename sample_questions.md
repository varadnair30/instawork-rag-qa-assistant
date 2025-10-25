### Test Case 1: The synonym and negative constraint query

This test evaluates the RAG's ability to understand synonyms ("creating an account" instead of "sign up" or "onboarding") and handle negative constraints ("don't have a resume").

-   **Query:**
    "How does a Pro create an account if they don't have a resume?"

-   **Expected result:**
    The RAG system should primarily retrieve the test case that explicitly details the onboarding flow *without* a resume. It should also recognize related but less relevant test cases.

    -   **Primary relevance (must be returned first):**
        -   **Testcase ID 25977:** "Verify Pro is able to complete onboarding flow without resume or positions selected in English". This is the most direct and accurate answer to the query.

    -   **Secondary relevance (good to return, but with lower priority):**
        -   **Testcase ID 25975:** "Verify Pro is able to complete onboarding flow with resume and positions selected in English". This is relevant as it describes the alternative path and provides context, but it's not the direct answer.
        -   **Testcase ID 25976:** "Verify Pro is able to complete onboarding flow in Spanish". Related onboarding flow test case that may also be returned.

### Test Case 2: The specific feature and language intersection query

This test assesses the system's precision in retrieving information based on multiple, specific keywords that define a narrow context (AI Coach Call + Spanish).

-   **Query:**
    "Show me test cases for the Spanish language option during the AI coach call."

-   **Expected result:**
    The system should identify test cases related to Spanish language support and coach calls, with preference for general Spanish support test cases appearing first.

    -   **Primary relevance (must be returned first):**
        -   **Testcase ID 2268:** "Verify App can be installed and navigated in Spanish support". This provides general Spanish language support validation across the app.
        -   **Testcase ID 25976:** "Verify Pro is able to complete onboarding flow in Spanish". This test case focuses on the entire onboarding flow in Spanish, contextually relevant to language-specific features.

    -   **Secondary relevance (good to return, but with lower priority):**
        -   **Testcase ID 9080:** "Verify Pro is required to schedule a human coach call...". This is relevant as it mentions selecting "English or Spanish" for a coach call (human coach call scenario).
        -   **Testcase ID 26676:** "Verify Pro can apply any available positions with AI coach call". This test case's first step describes starting a coach call and selecting a language, with the "Experience screening call" screen showing "Spanish" and "English" options.
        -   **Testcase ID 25975:** "Verify Pro is able to complete onboarding flow with resume and positions selected in English". Step 11 of this test case mentions the "Screening call required" screen where the Pro can tap on "Spanish" or "English".

### Test Case 3: The technical/error condition query

This query tests the RAG's ability to find information that isn't in a title but is buried within the "expected results" or "steps" of a test case, focusing on a specific error message.

-   **Query:**
    "What happens if a user tries to create an account with an invalid email address?"

-   **Expected result:**
    The system should retrieve validation-related test cases during account creation, with general validation test cases ranking higher.

    -   **Primary relevance (must be returned first):**
        -   **Testcase ID 26415:** "Validate birthdate and location steps during onboarding flow". While this focuses on birthdate and location validation, it represents the same validation error pattern during onboarding.
        -   **Testcase ID 25977:** "Verify Pro is able to complete onboarding flow without resume...". The "expected" field for the first step explicitly states: "Error messages if ... invalid email address" and includes an image attachment showing this error.

    -   **Secondary relevance (good to return, but with lower priority):**
        -   **Testcase ID 25884:** "Verify account gets deactivated with 5 violation points". This is related to account status and validation, though focused on post-creation account management.
        -   **Testcase ID 25975:** "Verify Pro is able to complete onboarding flow with resume...". This test case contains the exact same first step with email validation as TC25977, making it equally relevant for the specific error condition.

### Test Case 4: The vague concept / edge case query

This test is designed to see if the RAG can understand a user's intent even when they use vague, non-technical terms ("lose my internet connection") and connect it to a specific technical concept ("Offline mode").

-   **Query:**
    "What happens if I lose my internet connection while looking for shifts?"

-   **Expected result:**
    A strong RAG system will semantically link "lose internet connection" to "offline" and retrieve the relevant test cases from the "Offline mode" section.

    -   **Primary relevance (must be returned):**
        -   **Testcase ID 11076:** "Offline mode: Open Shifts first page is cached". This is the most direct answer as it describes exactly what happens on the "Open Shifts" screen during an internet outage.

    -   **Secondary relevance (good to return):**
        -   **Testcase ID 11096:** "Offline mode: Scrolling after coming back online (Open Shifts)". This is highly related, describing the behavior upon reconnection.
        -   **Testcase ID 11088:** "Offline mode: Error display for non-offline enabled screens". This provides broader context about how the app behaves offline on other screens.
        -   **Testcase ID 27696:** "Offline mode: Verify pro can clock in". While not about "looking for shifts", it's a key part of the offline functionality and demonstrates the RAG's ability to find related capabilities.

### Test Case 5: The plausible but non-existent feature (negative case)

This is a critical negative test case. It uses plausible, industry-standard terminology for a feature that does *not* exist in the provided data. This tests if the RAG avoids hallucination and correctly reports the absence of information.

-   **Query:**
    "Are there any test cases for dark mode or theme switching functionality?"

-   **Expected result:**
    The RAG system should return **no results** or a message explicitly stating that it could not find any test cases related to dark mode or theme switching.

    -   **Correct response:** "I could not find any test cases related to dark mode, theme switching, or appearance settings in the provided documentation."
    
    -   **Incorrect response (false positive):** Returning unrelated test cases would be a false positive. For example, if the system returns general UI test cases or shift-viewing test cases that don't actually test dark mode, this would prove the RAG system is over-generalizing and not paying attention to the specific constraints of the query.

    -   **Why this query:** Dark mode/theme switching is a common mobile app feature but is not present in any of the test cases. The app uses verification codes (not passwords), so queries about "password reset" might still match on "login" or "verification". Dark mode is specific enough to avoid semantic overlap with existing features.

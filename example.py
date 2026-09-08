"""
example.py - Comprehensive Test Scenarios for the ML LLM Router.

Run this file after running 'python train.py' to test all routing behaviors:
    python example.py
"""

import sys
import json

try:
    from app.router import predict_provider
except Exception as e:
    print(f"Error loading router: {e}")
    print("Please make sure you have run 'python train.py' first to generate model artifacts.")
    sys.exit(1)


# Define 6 distinct scenarios covering all providers and fallback mechanisms
TEST_CASES = [
    {
        "id": 1,
        "name": "Quick / Structured Task",
        "expected": "openai",
        "description": "Short, structured generation and utility tasks.",
        "query": "Format this list of customer names into a valid JSON object and write a quick regex to validate their emails."
    },
    {
        "id": 2,
        "name": "Long Context / Complex Architectural Analysis",
        "expected": "claude",
        "description": "Multi-step reasoning, long documents, and deep risk analysis.",
        "query": "Review this 50-page technical architecture document, evaluate microservices trade-offs, and identify potential race conditions."
    },
    {
        "id": 3,
        "name": "Multimodal / Visual Analysis",
        "expected": "gemini",
        "description": "Image, screenshot, infographic, and visual artifact inspection.",
        "query": "Look at this screenshot of our campaign dashboard and interpret the scatter plot chart outliers."
    },
    {
        "id": 4,
        "name": "Enterprise / Model Gateway Request",
        "expected": "litellm",
        "description": "Centrally managed, provider-agnostic, and abstract proxy requests.",
        "query": "Route this request through the organization's unified model gateway and central LLM proxy."
    },
    {
        "id": 5,
        "name": "Ambiguous / Low-Confidence Query (Triggers Fallback)",
        "expected": "litellm (Fallback Triggered)",
        "description": "Vague or casual query with no strong category features (< 0.60 confidence).",
        "query": "Hey, what do you think about going for a walk later today?"
    },
    {
        "id": 6,
        "name": "Novel Mixed Domain Query",
        "expected": "Evaluated by Learned Weights",
        "description": "Tests how the model handles cross-domain phrasing based on relative feature weights.",
        "query": "Analyze this marketing report and draft 5 variations of an email headline."
    }
]


def run_examples():
    print("=" * 80)
    print("       DYNAMIC ML LLM ROUTER - SCENARIO DEMONSTRATION")
    print("=" * 80)
    print()

    for case in TEST_CASES:
        print(f"Scenario {case['id']}: [{case['name']}]")
        print(f"Purpose:     {case['description']}")
        print(f"Input Query: \"{case['query']}\"")
        
        # Run inference through the ML Router
        result = predict_provider(case['query'])

        print("\n--- Model Output ---")
        print(f"  Selected Provider:   {result.provider.upper()}")
        print(f"  Confidence Score:    {result.confidence:.4f} ({result.confidence * 100:.1f}%)")
        print(f"  Fallback Triggered:  {result.fallback_triggered}")
        print("  Class Probabilities:")
        for provider, prob in result.probabilities.items():
            bar = "█" * int(prob * 25)
            print(f"    - {provider:<8}: {prob:.4f} | {bar}")

        print("-" * 80)
        print()


if __name__ == "__main__":
    run_examples()
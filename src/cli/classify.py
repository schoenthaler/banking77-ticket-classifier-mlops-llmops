"""
CLI tool for classifying customer tickets.
"""
import sys
import json
from src.guardrails.pipeline import process_ticket


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli.classify <text> [--llm] [--json]")
        print("Example: python -m src.cli.classify 'My account was hacked'")
        print("Options:")
        print("  --llm    Enable LLM-based guardrails (requires OPENAI_API_KEY)")
        print("  --json   Output results as JSON")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    use_llm = "--llm" in args
    output_json = "--json" in args
    
    # Remove flags from text
    text_args = [arg for arg in args if not arg.startswith("--")]
    text = " ".join(text_args)
    
    if not text:
        print("Error: No text provided")
        print("Usage: python -m src.cli.classify <text> [--llm] [--json]")
        sys.exit(1)
    
    print("="*80)
    print("Ticket Classification")
    print("="*80)
    print(f"Text: {text}\n")
    
    # Process ticket
    result = process_ticket(text, use_llm=use_llm)
    
    # JSON output
    if output_json:
        print(json.dumps(result, indent=2))
        return
    
    # Print results
    print("Results:")
    print("-" * 80)
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence: {result['model_confidence']:.4f}")
    print(f"Priority: {result['priority'].upper()}")
    print(f"Toxic: {'Yes' if result['is_toxic'] else 'No'}")
    print(f"Needs Manual Review: {'Yes' if result['needs_manual_review'] else 'No'}")
    
    if result['urgency']['urgent']:
        print(f"\n[!] URGENT: {result['urgency']['reason']}")
    
    if result['is_toxic']:
        print(f"\n[!] TOXIC: {result['toxicity']['reason']}")
    
    if result['needs_manual_review']:
        print(f"\n[!] REVIEW NEEDED: {result['validation']['reason']}")
    
    if result['llm_notes']:
        print(f"\nLLM Notes: {result['llm_notes']}")


if __name__ == "__main__":
    main()


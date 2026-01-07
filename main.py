"""
Main script for RAG Dual-Mode System

Usage:
    python main.py                  # Interactive menu
    python main.py --demo           # Run demo tests
    python main.py --evaluate       # Run full evaluation
    python main.py --chat           # Interactive chat mode

Modes:
- FLASH (⚡): Fast responses (~2-3s)
- THINKING (🧠): Deep verification (~10-20s)
"""

import argparse
import time
from src.rag_advanced import (
    get_rag_chain,
    get_flash_chain,
    get_thinking_chain,
    RAGMode,
    set_verbose,
    config  # Centralized configuration
)
from src.vector_store import get_smart_retriever
from rag_tests.test_utils import run_evaluation
from rag_tests.test import SAMPLE_DATASET

# Use centralized config for document retrieval
K_DOCS = config.K_DOCS_DEFAULT


def run_demo():
    """Run demonstration tests for both RAG modes."""
    print("Loading retriever...")
    retriever = get_smart_retriever(k_docs=K_DOCS)
    print("Using SmartRetriever with metadata boosting")

    # ============================================
    # TEST 1: Flash Mode (Fast)
    # ============================================
    print("\n" + "="*60)
    print("⚡ TEST 1: FLASH MODE - Basic query")
    print("="*60)

    set_verbose(enabled=False)
    flash_chain = get_flash_chain(retriever)

    question = "What are the security deposit requirements for interconnection?"
    print(f"Question: {question}")

    start = time.time()
    for chunk in flash_chain.stream(
        {"question": question},
        config={"configurable": {"session_id": "flash_test"}}
    ):
        print(chunk, end="", flush=True)
    flash_time = time.time() - start
    print(f"\n\n⏱️ Flash mode time: {flash_time:.2f}s")

    # ============================================
    # TEST 2: Thinking Mode (Deep Verification)
    # ============================================
    print("\n\n" + "="*60)
    print("🧠 TEST 2: THINKING MODE - Same query")
    print("="*60)

    thinking_chain = get_thinking_chain(retriever)

    print(f"Question: {question}")

    start = time.time()
    for chunk in thinking_chain.stream(
        {"question": question},
        config={"configurable": {"session_id": "thinking_test"}}
    ):
        print(chunk, end="", flush=True)
    thinking_time = time.time() - start
    print(f"\n\n⏱️ Thinking mode time: {thinking_time:.2f}s")

    # ============================================
    # TEST 3: Thinking Mode - Spanish
    # ============================================
    print("\n\n" + "="*60)
    print("🧠 TEST 3: THINKING MODE - Spanish query")
    print("="*60)

    question_es = "Cuáles son los requisitos de seguridad para proyectos solares?"
    print(f"Question: {question_es}")

    start = time.time()
    for chunk in thinking_chain.stream(
        {"question": question_es},
        config={"configurable": {"session_id": "thinking_es"}}
    ):
        print(chunk, end="", flush=True)
    print(f"\n\n⏱️ Thinking mode time: {time.time() - start:.2f}s")

    # ============================================
    # TEST 4: Out-of-scope question
    # ============================================
    print("\n\n" + "="*60)
    print("⚡ TEST 4: FLASH MODE - Out-of-scope question")
    print("="*60)

    question_oos = "What is the capital of France?"
    print(f"Question: {question_oos}")

    for chunk in flash_chain.stream(
        {"question": question_oos},
        config={"configurable": {"session_id": "oos_test"}}
    ):
        print(chunk, end="", flush=True)

    # ============================================
    # SUMMARY
    # ============================================
    print("\n\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Flash mode:    {flash_time:.2f}s")
    print(f"Thinking mode: {thinking_time:.2f}s")
    print(f"Overhead:      {thinking_time - flash_time:.2f}s ({(thinking_time/flash_time - 1)*100:.0f}% slower)")
    print("="*60)
    print("\nALL TESTS COMPLETE ✓")


def run_chat():
    """Run interactive chat mode."""
    print("Loading retriever...")
    retriever = get_smart_retriever(k_docs=K_DOCS)

    print("\n" + "="*60)
    print("💬 INTERACTIVE CHAT MODE")
    print("="*60)
    print("Commands:")
    print("  /flash    - Switch to Flash mode (⚡)")
    print("  /thinking - Switch to Thinking mode (🧠)")
    print("  /verbose  - Toggle verbose output")
    print("  /summary  - Toggle auto-summary of sources")
    print("  /clear    - Clear chat history")
    print("  /exit     - Exit chat")
    print("="*60)

    current_mode = RAGMode.THINKING
    verbose = True
    with_summary = True
    session_id = "chat_session"

    # Enable verbose by default
    set_verbose(enabled=True)

    # Build initial chain (Thinking mode with summary)
    chain = get_thinking_chain(retriever, with_summary=with_summary)

    while True:
        try:
            mode_indicator = f"{current_mode.value}"
            if with_summary:
                mode_indicator += "+summary"
            user_input = input(f"\n[{mode_indicator}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # Command handling
        if user_input.lower() == "/exit":
            # Clear history on exit
            from src.rag_advanced import get_session_history
            get_session_history(session_id).clear()
            print("\nGoodbye! 👋")
            break
        elif user_input.lower() == "/clear":
            from src.rag_advanced import get_session_history
            get_session_history(session_id).clear()
            print("🧹 Chat history cleared")
            continue
        elif user_input.lower() == "/flash":
            current_mode = RAGMode.FLASH
            chain = get_flash_chain(retriever, with_summary=with_summary)
            print("⚡ Switched to Flash mode")
            continue
        elif user_input.lower() == "/thinking":
            current_mode = RAGMode.THINKING
            chain = get_thinking_chain(retriever, with_summary=with_summary)
            print("🧠 Switched to Thinking mode")
            continue
        elif user_input.lower() == "/verbose":
            verbose = not verbose
            set_verbose(enabled=verbose)
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue
        elif user_input.lower() == "/summary":
            with_summary = not with_summary
            # Rebuild chain with new summary setting
            if current_mode == RAGMode.FLASH:
                chain = get_flash_chain(retriever, with_summary=with_summary)
            else:
                chain = get_thinking_chain(retriever, with_summary=with_summary)
            print(f"📝 Auto-summary: {'ON' if with_summary else 'OFF'}")
            continue

        # Get response
        print("\nAssistant: ", end="", flush=True)
        start = time.time()
        response_chunks = []
        for chunk in chain.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        ):
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)

        # Check if this was a rejection - if so, clear this exchange from history
        full_response = "".join(response_chunks)
        if "not related to ERCOT" in full_response or "no está relacionada" in full_response:
            # Clear last exchange from session history to avoid polluting context
            from src.rag_advanced import get_session_history
            history = get_session_history(session_id)
            if len(history.messages) >= 2:
                # Remove last human + AI message pair
                history.messages = history.messages[:-2]

        print(f"\n\n⏱️ {time.time() - start:.2f}s")


def show_menu():
    """Show interactive menu."""
    print("\n" + "="*60)
    print("🤖 RAG DUAL-MODE SYSTEM")
    print("="*60)
    print("\nSelect an option:")
    print("  1. Run Demo Tests")
    print("  2. Run Full Evaluation")
    print("  3. Interactive Chat")
    print("  4. Exit")
    print("="*60)

    while True:
        try:
            choice = input("\nChoice [1-4]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            return

        if choice == "1":
            run_demo()
        elif choice == "2":
            run_evaluation(SAMPLE_DATASET, mode=RAGMode.THINKING)
        elif choice == "3":
            run_chat()
        elif choice == "4":
            print("\nGoodbye! 👋")
            return
        else:
            print("Invalid choice. Please enter 1-4.")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="RAG Dual-Mode System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Interactive menu
    python main.py --demo       # Run demo tests
    python main.py --evaluate   # Run full evaluation
    python main.py --chat       # Interactive chat mode
        """
    )
    parser.add_argument("--demo", action="store_true", help="Run demo tests")
    parser.add_argument("--evaluate", action="store_true", help="Run full evaluation")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.evaluate:
        run_evaluation(SAMPLE_DATASET, mode=RAGMode.THINKING)
    elif args.chat:
        run_chat()
    else:
        show_menu()


if __name__ == "__main__":
    main()

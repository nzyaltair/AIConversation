"""
Qwen3.5 Chat - Interactive chat test script
Usage: python run_chat.py [--no-think] [--backend vulkan] [--verbose]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import ChatEngine, ChatConfig


def safe_print(text: str):
    """Print text safely, handling surrogate characters."""
    try:
        print(text, end="", flush=True)
    except UnicodeEncodeError:
        # Fallback: replace unencodable chars
        print(text.encode(sys.stdout.encoding or "utf-8", errors="replace")
                  .decode(sys.stdout.encoding or "utf-8", errors="replace"),
              end="", flush=True)


def parse_args():
    # Resolve default model_dir relative to this script's location, not CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.abspath(os.path.join(script_dir, "..", "models", "Qwen3.5-0.8B.Q4_K_M"))
    args = {
        "backend": "auto",
        "thinking": True,
        "verbose": False,
        "model_dir": default_model_dir,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--no-think":
            args["thinking"] = False
        elif argv[i] == "--backend" and i + 1 < len(argv):
            args["backend"] = argv[i + 1]
            i += 1
        elif argv[i] == "--verbose":
            args["verbose"] = True
        elif argv[i] == "--temp" and i + 1 < len(argv):
            args["temperature"] = float(argv[i + 1])
            i += 1
        elif argv[i] == "--max-tokens" and i + 1 < len(argv):
            args["max_tokens"] = int(argv[i + 1])
            i += 1
        elif argv[i] == "--model-dir" and i + 1 < len(argv):
            args["model_dir"] = argv[i + 1]
            i += 1
        elif argv[i] in ("-h", "--help"):
            print(__doc__)
            print("Options:")
            print("  --no-think       Disable thinking mode (faster, direct answer)")
            print("  --backend TYPE   GPU backend: auto / vulkan / cuda / cpu")
            print("  --verbose        Show detailed llama.cpp logs")
            print("  --temp FLOAT     Temperature (default 0.7)")
            print("  --max-tokens N   Max tokens to generate (default 512)")
            print("  --model-dir DIR  Model directory")
            sys.exit(0)
        i += 1
    return args


def main():
    args = parse_args()

    print("=" * 55)
    print("  Qwen3.5-0.8B Chat - llama.cpp Inference")
    print("=" * 55)

    config = ChatConfig(
        model_dir=args["model_dir"],
        llm_backend=args["backend"],
        enable_thinking=args["thinking"],
        verbose=args["verbose"],
    )
    print(f"Loading model (backend={args['backend']}, "
          f"thinking={'ON' if args['thinking'] else 'OFF'})...")

    engine = ChatEngine(config)

    print(f"Backend: {engine.active_backend}")
    print(f"Available: {engine.available_backends}")
    print(f"Thinking: {'ON' if args['thinking'] else 'OFF'}")
    print()

    # Warmup
    print("Warming up...", end=" ", flush=True)
    _ = engine.chat(prompt="Hi", max_tokens=4)
    print("Done!\n")

    print("Commands: /think | /nothink | /stats | /clear | /exit")
    print("-" * 55)

    thinking = args["thinking"]

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Bye!")
            break
        elif user_input == "/think":
            thinking = True
            print("[Thinking: ON]")
            continue
        elif user_input == "/nothink":
            thinking = False
            print("[Thinking: OFF]")
            continue
        elif user_input == "/stats":
            s = engine.last_stats
            if s:
                print(f"  Backend: {s.get('backend')}  |  Thinking: {s.get('thinking')}")
                print(f"  Prefill: {s.get('n_prefill')}t @ {s.get('prefill_speed_tps')}t/s "
                      f"({s.get('t_prefill_s')}s)")
                print(f"  Generate: {s.get('n_generate')}t @ {s.get('gen_speed_tps')}t/s "
                      f"({s.get('t_generate_s')}s)")
                print(f"  TTFT: {s.get('ttft_ms')}ms  |  Total: {s.get('t_total_s')}s")
            else:
                print("  No stats yet. Chat first.")
            continue
        elif user_input == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # Generate
        safe_print("AI  > ")
        try:
            for chunk in engine.stream_chat(
                prompt=user_input,
                temperature=args["temperature"],
                max_tokens=args["max_tokens"],
                enable_thinking=thinking,
            ):
                safe_print(chunk)
            print()
        except Exception as e:
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    main()

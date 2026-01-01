"""Verify project structure and syntax without dependencies."""

import ast
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


def check_python_syntax(file_path: Path) -> tuple[bool, str]:
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Check all new Python files."""
    print("Verifying project structure...\n")

    # Define new files to check
    new_files = [
        # Database
        "src/database/schema_sqlite.py",
        "src/database/connection_sqlite.py",

        # Memory
        "src/memory/vector_store_adapter.py",
        "src/memory/session_memory_service.py",

        # LLM
        "src/llm/provider_abstraction.py",

        # Search workflow
        "src/workflow/search/__init__.py",
        "src/workflow/search/classifier.py",
        "src/workflow/search/actions.py",
        "src/workflow/search/researcher.py",
        "src/workflow/search/writer.py",
        "src/workflow/search/service.py",

        # Research workflow
        "src/workflow/research/__init__.py",
        "src/workflow/research/state.py",
        "src/workflow/research/queue.py",
        "src/workflow/research/nodes.py",
        "src/workflow/research/researcher.py",
        "src/workflow/research/graph.py",

        # Scripts
        "scripts/migrate_to_sqlite.py",

        # Tests
        "tests/__init__.py",
        "tests/integration/__init__.py",
        "tests/integration/test_basic_integration.py",
    ]

    all_ok = True
    stats = {"ok": 0, "missing": 0, "errors": 0}

    for file_rel_path in new_files:
        file_path = backend_path / file_rel_path

        if not file_path.exists():
            print(f"✗ MISSING: {file_rel_path}")
            stats["missing"] += 1
            all_ok = False
            continue

        ok, message = check_python_syntax(file_path)

        if ok:
            print(f"✓ {file_rel_path}")
            stats["ok"] += 1
        else:
            print(f"✗ {file_rel_path}: {message}")
            stats["errors"] += 1
            all_ok = False

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  ✓ Valid files: {stats['ok']}")
    print(f"  ✗ Missing files: {stats['missing']}")
    print(f"  ✗ Syntax errors: {stats['errors']}")
    print("=" * 70)

    if all_ok:
        print("\n✅ All files verified successfully!")
        return 0
    else:
        print("\n❌ Some files have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())

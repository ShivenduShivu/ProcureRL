import importlib.util
import inspect
import pathlib
import sys
import traceback


def _load_module(path_str):
    path = pathlib.Path(path_str)
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iter_test_functions(module):
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            yield name, obj


def main():
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if not args:
        print("usage: python -m pytest <test_file>")
        raise SystemExit(2)

    overall_passed = 0
    overall_failed = 0

    for test_path in args:
        module = _load_module(test_path)
        print(f"collecting tests from {test_path}")
        for test_name, test_fn in _iter_test_functions(module):
            try:
                test_fn()
                overall_passed += 1
                print(f"{test_name} PASSED")
            except Exception:
                overall_failed += 1
                print(f"{test_name} FAILED")
                traceback.print_exc()

    print(f"{overall_passed + overall_failed} tests collected")
    print(f"{overall_passed} passed, {overall_failed} failed")
    raise SystemExit(1 if overall_failed else 0)


if __name__ == "__main__":
    main()

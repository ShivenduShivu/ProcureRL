import sys

from .server import _load_target, run


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m uvicorn module:app --host 0.0.0.0 --port 7860")

    target = sys.argv[1]
    host = "127.0.0.1"
    port = 8000

    args = sys.argv[2:]
    for index, arg in enumerate(args):
        if arg == "--host" and index + 1 < len(args):
            host = args[index + 1]
        if arg == "--port" and index + 1 < len(args):
            port = int(args[index + 1])

    app = _load_target(target)
    run(app, host=host, port=port)


if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="CLI para levantar la app Streamlit.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--app-path", type=str, default="app/app.py")
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        args.app_path,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()


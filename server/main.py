import uvicorn

from .environment import app

def run() -> None:
    uvicorn.run(app, host="0.0.0.0", port=7860)

def main() -> None:
    run()

if __name__ == "__main__":
    run()

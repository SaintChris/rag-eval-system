import subprocess
import sys
import os


def main():
    print("Starting MLflow Tracking Server on port 5000...")
    mlflow_env = os.environ.copy()
    mlflow_env["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
    mlflow_process = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "server",
         "--host", "0.0.0.0", "--port", "5000",
         "--backend-store-uri", "sqlite:///mlflow.db",
         "--default-artifact-root", "./mlruns"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=mlflow_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print("Starting FastAPI Backend on port 8000...")
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "app:app",
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
    except KeyboardInterrupt:
        print("\nShutting down servers...")
    finally:
        mlflow_process.terminate()
        mlflow_process.wait()


if __name__ == "__main__":
    main()

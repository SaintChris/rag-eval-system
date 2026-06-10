import mlflow


class MLflowTracker:
    def __init__(self, experiment_name="rag-eval-system", tracking_uri="sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.create_experiment(self.experiment_name)
        except Exception:
            pass
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str):
        mlflow.log_artifact(file_path)

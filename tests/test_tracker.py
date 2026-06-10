import mlflow
from backend.tracker import MLflowTracker


def test_tracker_logging():
    tracker = MLflowTracker(experiment_name="test-rag-experiment")
    with tracker.start_run(run_name="test_run") as run:
        tracker.log_params({
            "chunk_size": 500,
            "chunk_overlap": 50,
            "model": "qwen3:4b"
        })
        tracker.log_metrics({
            "avg_faithfulness": 4.5,
            "avg_relevance": 4.8,
            "retrieval_accuracy": 0.90
        })
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("test-rag-experiment")
    assert exp is not None

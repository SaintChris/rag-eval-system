from eval.evaluator import RAGEvaluator


def test_eval_metrics():
    evaluator = RAGEvaluator(judge_model="qwen3:1.7b")
    context = "Alex is an MLOps Engineer who built a 6-agent system on his M1 Mac."
    answer = "Alex built a 6-agent system on an M1 Mac as an MLOps Engineer."
    reference = "Alex is an MLOps Engineer who created a 6-agent system using local M1 Mac resources."
    faithfulness = evaluator.score_faithfulness(context, answer)
    relevance = evaluator.score_relevance("What did Alex build?", answer, reference)
    assert 1.0 <= faithfulness <= 5.0
    assert 1.0 <= relevance <= 5.0

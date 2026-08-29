import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import subprocess

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Q4_Reproducibility")

seed = 42

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
except Exception:
    git_commit = "unknown"

with mlflow.start_run(run_name="Partner_A_Run") as run:
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=seed, max_iter=200).fit(X, y)
    accuracy = clf.score(X, y)
    
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("random_seed", seed)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("git_commit", git_commit)
    
    mlflow.sklearn.log_model(
        clf, 
        "model",
        registered_model_name="IrisLogisticRegression"
    )
    
    print(f"Partner A run completed with accuracy: {accuracy}")
    print(f"Logged Git Commit: {git_commit}")

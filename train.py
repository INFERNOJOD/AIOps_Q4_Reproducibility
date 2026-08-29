import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Q4_Reproducibility")

with mlflow.start_run(run_name="Partner_A_Run"):
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=42, max_iter=200).fit(X, y)
    accuracy = clf.score(X, y)
    
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    print(f"Partner A run completed with accuracy: {accuracy}")

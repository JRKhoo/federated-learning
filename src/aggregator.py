import numpy as np
from pathlib import Path
from model_tester import ModelTester
import pandas as pd

def evaluate_all_clients():
    clients = ["hospital1", "hospital2", "hospital3"]
    results = []

    for client in clients:
        weights_file = f"global_model/{client}_weights.npz"
        data_file = f"data/split/{client}.csv"

        print(f"\n=== Evaluating {client} ===")
        tester = ModelTester()
        tester.load_weights(weights_file)
        metrics = tester.evaluate_model(data_file)
        metrics["client"] = client

        results.append(metrics)

        

    # convert to DataFrame for readability
    df = pd.DataFrame(results)

    

    print("\n\n===== Summary Across Hospitals =====")

    print(df[["client", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]])
    print("\nAverages:")
    print(df.mean(numeric_only=True))
    print("\nStd Dev:")
    print(df.std(numeric_only=True))

if __name__ == "__main__":
    evaluate_all_clients()

import sys
from pathlib import Path
import numpy as np
from typing import List

from trainer import FederatedTrainer
from model_tester import ModelTester
import config.dp_config as dp_config
import csv


def find_hospital_csvs(split_dir: Path) -> List[Path]:
    files = sorted(split_dir.glob("hospital*.csv"))
    return files


def average_weights(list_of_weight_lists: List[List[np.ndarray]]) -> List[np.ndarray]:
    # assume every entry is same-length list of numpy arrays
    num_clients = len(list_of_weight_lists)
    num_arrays = len(list_of_weight_lists[0])
    aggregated = []
    for i in range(num_arrays):
        stacked = np.stack([client[i] for client in list_of_weight_lists], axis=0)
        aggregated.append(np.mean(stacked, axis=0))
    return aggregated


def zeros_like_weights(weights_template: List[np.ndarray]) -> List[np.ndarray]:
    return [np.zeros_like(w) for w in weights_template]


def run(rounds: int = 50, evaluate_every: int = 10, local_epochs: int = 1):
    base = Path(__file__).resolve().parents[1]
    split_dir = base / "data" / "split"
    weights_dir = base / "weights"
    weights_dir.mkdir(exist_ok=True)

    hospital_csvs = find_hospital_csvs(split_dir)
    if not hospital_csvs:
        print(f"No hospital CSVs found in {split_dir}")
        sys.exit(1)

    print(f"Found {len(hospital_csvs)} hospitals:")
    for p in hospital_csvs:
        print(f"  - {p}")

    # determine per-round epsilon based on config
    if getattr(dp_config, "AUTO_DISTRIBUTE", False) and getattr(dp_config, "TOTAL_EPSILON", None) is not None:
        per_round_epsilon = float(dp_config.TOTAL_EPSILON) / float(rounds)
        print(f"AUTO_DISTRIBUTE enabled: TOTAL_EPSILON={dp_config.TOTAL_EPSILON}, per-round epsilon={per_round_epsilon}")
    else:
        per_round_epsilon = float(dp_config.EPSILON)

    # prepare metrics CSV
    metrics_csv = weights_dir / "metrics.csv"
    write_header = not metrics_csv.exists()
    metrics_file = open(metrics_csv, "a", newline="")
    csv_writer = csv.writer(metrics_file)
    if write_header:
        csv_writer.writerow(["round", "per_round_epsilon", "sigma", "total_samples", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "num_samples"])

    # global model weights (interleaved list) - start as None
    global_weights = None

    # compute sample counts per client (used for weighted aggregation)
    sample_counts = []
    for p in hospital_csvs:
        try:
            # count rows excluding header
            with open(p, 'r') as fh:
                n = sum(1 for _ in fh) - 1
        except Exception:
            n = 0
        sample_counts.append(max(n, 0))

    total_samples = sum(sample_counts) if sum(sample_counts) > 0 else len(hospital_csvs)

    # create trainer per client (reused) to keep code simple
    trainers = [FederatedTrainer() for _ in hospital_csvs]

    # compute normalized client weights for aggregation
    client_weights = [c / total_samples for c in sample_counts]

    for r in range(1, rounds + 1):
        print(f"\n=== Round {r}/{rounds} ===")

        local_noisy_weights = []

        for i, csv_path in enumerate(hospital_csvs):
            trainer = trainers[i]

            # if no global weights yet, pass None so trainer initializes from scratch
            initial = None if global_weights is None else global_weights

            local_trained = trainer.train_from(str(csv_path), initial_weights=initial, epochs=local_epochs)

            # original weights for DP calculation
            if global_weights is None:
                orig = zeros_like_weights(local_trained)
            else:
                orig = global_weights

            noisy = trainer.add_dp_noise(local_trained, orig, epsilon=per_round_epsilon, delta=dp_config.DELTA, clip_norm=dp_config.CLIP_NORM)
            local_noisy_weights.append(noisy)

        # compute noise scale (sigma) for logging
        clip_norm = float(dp_config.CLIP_NORM)
        delta = float(dp_config.DELTA)
        # avoid division by zero
        eps = max(1e-12, float(per_round_epsilon))
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / eps

        # weighted aggregation by client sample counts
        # list_of_weight_lists: List[List[np.ndarray]] where outer list over clients
        num_arrays = len(local_noisy_weights[0])
        aggregated = []
        for arr_idx in range(num_arrays):
            accum = None
            for client_idx, client_weights_list in enumerate(local_noisy_weights):
                w = client_weights[client_idx]
                arr = client_weights_list[arr_idx]
                if accum is None:
                    accum = arr * w
                else:
                    accum = accum + arr * w
            aggregated.append(accum)
        global_weights = aggregated

        # save global weights for this round
        out_file = weights_dir / f"global_round_{r}.npz"
        np.savez(out_file, *global_weights)
        # also update canonical global.npz
        np.savez(weights_dir / "global.npz", *global_weights)

        print(f"Saved aggregated global model for round {r} to {out_file}")

        # periodic evaluation
        metrics = None
        if r % evaluate_every == 0 or r == rounds:
            print(f"Evaluating global model at round {r}...")
            tester = ModelTester()
            tester.load_weights(str(weights_dir / "global.npz"))
            metrics = tester.evaluate_model(str(base / "data" / "split" / "test_data.csv"))

        # write metrics row (metrics may be None if not evaluated this round)
        if metrics is None:
            csv_writer.writerow([r, per_round_epsilon, sigma, total_samples, "", "", "", "", "", "", ""])
        else:
            csv_writer.writerow([
                r,
                per_round_epsilon,
                sigma,
                total_samples,
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("roc_auc"),
                metrics.get("pr_auc"),
                metrics.get("num_samples"),
            ])
        metrics_file.flush()

    metrics_file.close()


def main():
    rounds = 50
    evaluate_every = 10
    local_epochs = 1

    # allow CLI overrides: rounds, evaluate_every, local_epochs
    if len(sys.argv) > 1:
        try:
            rounds = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            evaluate_every = int(sys.argv[2])
        except ValueError:
            pass
    if len(sys.argv) > 3:
        try:
            local_epochs = int(sys.argv[3])
        except ValueError:
            pass

    run(rounds=rounds, evaluate_every=evaluate_every, local_epochs=local_epochs)


if __name__ == "__main__":
    main()

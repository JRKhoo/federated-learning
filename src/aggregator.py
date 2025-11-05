import sys
from pathlib import Path
import numpy as np
from typing import List, Optional, Tuple

import config.dp_config as dp_config
from model_tester import ModelTester


class FederatedAggregator:
    
    def __init__(self):
        self.global_weights = None
    
    def load_hospital_weights(self, weights_file: str) -> List[np.ndarray]:
        print(f"Loading weights from: {weights_file}")
        data = np.load(weights_file)
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        return weights
    
    def add_dp_noise(self, weights: List[np.ndarray], base_weights: Optional[List[np.ndarray]] = None, *, epsilon: Optional[float] = None, delta: Optional[float] = None, clip_norm: Optional[float] = None) -> List[np.ndarray]:
        """
        Add DP noise to aggregated weights using the Gaussian mechanism.
        If base_weights is provided, treats input as updates and clips before adding noise.
        """
        # use config values if not overridden
        if epsilon is None:
            epsilon = dp_config.EPSILON
        if delta is None:
            delta = dp_config.DELTA
        if clip_norm is None:
            clip_norm = dp_config.CLIP_NORM
            
        print(f"\nApplying central differential privacy...")
        print(f"Parameters: CLIP_NORM={clip_norm}, EPSILON={epsilon}, DELTA={delta}")
        
        if base_weights is not None:
            # Treat as updates: compute and clip
            updates = [w - w_base for w, w_base in zip(weights, base_weights)]
            global_norm = np.sqrt(sum(np.sum(u ** 2) for u in updates))
            clip_factor = min(1.0, float(clip_norm) / (global_norm + 1e-10))
            weights = [w_base + (u * clip_factor) for w_base, u in zip(base_weights, updates)]
            
            print(f"Update L2 norm: {global_norm:.4f}")
            print(f"Clip factor: {clip_factor:.4f}")
        
        # Add Gaussian noise scaled by sensitivity/epsilon
        sigma = clip_norm * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        print(f"Noise scale (sigma): {sigma:.4f}")
        
        noisy_weights = []
        for i, w in enumerate(weights):
            noise = np.random.normal(0, sigma, w.shape)
            noisy_w = w + noise
            noisy_weights.append(noisy_w)
            
            weight_type = "Weights" if i % 2 == 0 else "Biases"
            layer = i // 2 + 1
            noise_norm = np.sqrt(np.sum(noise ** 2))
            print(f"Layer {layer} {weight_type} noise L2 norm: {noise_norm:.4f}")
            
        return noisy_weights

    def aggregate_weights(self, weight_files: List[str], base_weights: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        if not weight_files:
            raise ValueError("No weight files provided for aggregation")
        
        print("\nAggregating weights...")
        
        # load weights
        all_hospital_weights = []
        for weight_file in weight_files:
            weights = self.load_hospital_weights(weight_file)
            all_hospital_weights.append(weights)
        
        # verify all hospitals use same training architecture
        num_layers = len(all_hospital_weights[0])
        for i, hospital_weights in enumerate(all_hospital_weights):
            if len(hospital_weights) != num_layers:
                raise ValueError(f"Hospital {i} has different number of layers")
        
        # compute average weights for each layer
        aggregated_weights = []
        for layer_idx in range(num_layers):
            layer_weights = [hospital[layer_idx] for hospital in all_hospital_weights]
            avg_weight = np.mean(layer_weights, axis=0)
            aggregated_weights.append(avg_weight)
            
            weight_type = "Weights" if layer_idx % 2 == 0 else "Biases"
            print(f"  Layer {layer_idx//2 + 1} {weight_type}: shape {avg_weight.shape}")
        
        # Apply DP noise after aggregation
        self.global_weights = self.add_dp_noise(aggregated_weights, base_weights)
        print(f"\nAggregation complete")
        
        return aggregated_weights
    
    # save global model weights to file
    def save_global_model(self, output_file: str) -> None:
        if self.global_weights is None:
            raise ValueError("No global weights to save. Run aggregate_weights first.")
        
        np.savez(output_file, *self.global_weights)
        print(f"\nGlobal model saved to: {output_file}")


def main():
    weights_dir = Path("weights")

    if not weights_dir.exists():
        print(f"Error: Directory '{weights_dir}' not found")
        sys.exit(1)
    
    # find all individual hospital weights in the directory
    weight_files = list(weights_dir.glob("*_weights.npz"))

    if not weight_files:
        print(f"Error: No weight files found in '{weights_dir}'")
        print("Weight files should have the pattern: *_weights.npz")
        sys.exit(1)
    
    # convert to strings for processing
    weight_file_paths = [str(f) for f in weight_files]
    
    print(f"Found {len(weight_file_paths)} hospital weight files:")
    for f in weight_file_paths:
        print(f"  - {f}")
    
    # initialize aggregator
    aggregator = FederatedAggregator()
    
    # aggregate weights from all hospitals
    aggregator.aggregate_weights(weight_file_paths)
    
    # save global model
    output_file = "weights/global.npz"
    aggregator.save_global_model(output_file)
    
    # evaluate global model on test data
    print("\nEvaluating Global Model")
    tester = ModelTester()
    tester.load_weights(output_file)
    tester.evaluate_model("data/split/test_data.csv")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import numpy as np
from typing import List

from model_tester import ModelTester


class FederatedAggregator:
    
    def __init__(self):
        self.global_weights = None
    
    def load_hospital_weights(self, weights_file: str) -> List[np.ndarray]:
        print(f"Loading weights from: {weights_file}")
        data = np.load(weights_file)
        weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        return weights
    
    def aggregate_weights(self, weight_files: List[str]) -> List[np.ndarray]:
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
        
        self.global_weights = aggregated_weights
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

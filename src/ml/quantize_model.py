import os
import time
import torch
import mlflow
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from torch.quantization import quantize_dynamic
import numpy as np

from src.data.label_mapping import get_label_mappings


def get_model_size_mb(model_path):
    """Calculate model size in MB."""
    total_size = 0
    if os.path.isdir(model_path):
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    else:
        total_size = os.path.getsize(model_path)
    return total_size / (1024 * 1024)  # Convert to MB


def benchmark_inference(model, tokenizer, test_texts, num_runs=100):
    """
    Benchmark inference latency.
    
    Args:
        model: The model to benchmark
        tokenizer: Tokenizer
        test_texts: List of test texts
        num_runs: Number of inference runs
        
    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    
    # Tokenize test texts
    inputs = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(**inputs)
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
    }


def quantize_distilbert(
    model_path="models/distilbert-banking77",
    output_path="models/quantized/distilbert-banking77",
    log_to_mlflow=True,
    experiment_name="banking77-distilbert"
):
    """
    Quantize a trained DistilBERT model using dynamic quantization.
    
    Args:
        model_path: Path to the trained model
        output_path: Path to save quantized model
        log_to_mlflow: Whether to log to MLflow
        experiment_name: MLflow experiment name
    """
    print("="*80)
    print("DistilBERT Model Quantization")
    print("="*80)
    
    # Load original model
    print(f"\n1. Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get model info
    id_to_label, label_to_id = get_label_mappings()
    num_labels = len(id_to_label)
    
    # Calculate original model size
    original_size_mb = get_model_size_mb(model_path)
    print(f"   Original model size: {original_size_mb:.2f} MB")
    
    # Prepare model for quantization
    print("\n2. Preparing model for quantization...")
    model.eval()
    
    # Apply dynamic quantization (int8 for linear layers)
    print("\n3. Applying dynamic quantization (int8)...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8
    )
    
    print("   ✓ Quantization complete")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save quantized model
    print(f"\n4. Saving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save entire quantized model object (quantized models have different structure)
    quantized_model_path = os.path.join(output_path, "quantized_model.pt")
    torch.save(quantized_model, quantized_model_path)
    print(f"   ✓ Saved quantized model")
    
    # Save config
    model.config.save_pretrained(output_path)
    print(f"   ✓ Saved model config")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"   ✓ Saved tokenizer")
    
    # Save quantization info
    import json
    quantization_info = {
        "quantization_method": "dynamic",
        "quantization_dtype": "int8",
        "quantized_layers": "Linear"
    }
    with open(os.path.join(output_path, "quantization_info.json"), "w") as f:
        json.dump(quantization_info, f, indent=2)
    
    # Calculate quantized model size
    quantized_size_mb = get_model_size_mb(output_path)
    size_reduction = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100
    print(f"   Quantized model size: {quantized_size_mb:.2f} MB")
    print(f"   Size reduction: {size_reduction:.1f}%")
    
    # Benchmark original model
    print("\n5. Benchmarking original model...")
    test_texts = [
        "How do I activate my card?",
        "What is my account balance?",
        "I need to change my PIN",
        "Can I get a refund?",
        "How do I transfer money?"
    ]
    
    original_stats = benchmark_inference(model, tokenizer, test_texts)
    print(f"   Mean latency: {original_stats['mean_latency_ms']:.2f} ms")
    
    # Benchmark quantized model
    print("\n6. Benchmarking quantized model...")
    quantized_stats = benchmark_inference(quantized_model, tokenizer, test_texts)
    print(f"   Mean latency: {quantized_stats['mean_latency_ms']:.2f} ms")
    
    speedup = original_stats['mean_latency_ms'] / quantized_stats['mean_latency_ms']
    print(f"   Speedup: {speedup:.2f}x")
    
    # Log to MLflow
    if log_to_mlflow:
        print("\n7. Logging to MLflow...")
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="Quantized Model - int8"):
            # Log quantization parameters
            mlflow.log_params({
                "quantization_method": "dynamic",
                "quantization_dtype": "int8",
                "quantized_layers": "Linear",
                "original_model_path": model_path,
                "num_labels": num_labels
            })
            
            # Log model metrics
            mlflow.log_metrics({
                "original_size_mb": original_size_mb,
                "quantized_size_mb": quantized_size_mb,
                "size_reduction_percent": size_reduction,
                "original_mean_latency_ms": original_stats['mean_latency_ms'],
                "quantized_mean_latency_ms": quantized_stats['mean_latency_ms'],
                "speedup_factor": speedup,
                "original_p95_latency_ms": original_stats['p95_latency_ms'],
                "quantized_p95_latency_ms": quantized_stats['p95_latency_ms'],
            })
            
            # Log quantized model as artifact
            mlflow.log_artifacts(output_path, artifact_path="quantized_model")
            
            print(f"   ✓ Logged to MLflow (Run ID: {mlflow.active_run().info.run_id})")
    
    print("\n" + "="*80)
    print("Quantization Complete!")
    print("="*80)
    print(f"Original size: {original_size_mb:.2f} MB")
    print(f"Quantized size: {quantized_size_mb:.2f} MB")
    print(f"Size reduction: {size_reduction:.1f}%")
    print(f"Speedup: {speedup:.2f}x")
    print(f"\nQuantized model saved to: {output_path}")
    
    return quantized_model, {
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "size_reduction": size_reduction,
        "speedup": speedup,
        "original_stats": original_stats,
        "quantized_stats": quantized_stats
    }


if __name__ == "__main__":
    quantize_distilbert()
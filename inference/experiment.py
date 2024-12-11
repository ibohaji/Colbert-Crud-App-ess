from dataclasses import dataclass
from typing import Dict, List
import time
import psutil
import json
import os
import platform
import torch
from datetime import datetime
from pathlib import Path

@dataclass
class QueryExperiment:
    query_id: str
    latency: float  # seconds
    memory_used: float  # MB
    
@dataclass
class ModelExperiment:
    model_name: str
    device: str
    model_dimension: int
    index_size_mb: float
    model_size_mb: float
    load_time: float
    warmup_time: float
    index_time: float
    query_experiments: List[QueryExperiment]
    retrieval_model: str = None
    
    def calculate_average_latency(self) -> float:
        """Calculate average latency across all queries"""
        if not self.query_experiments:
            return 0.0
        return sum(qe.latency for qe in self.query_experiments) / len(self.query_experiments)
    
    def calculate_throughput(self) -> float:
        """Calculate queries per second"""
        if not self.query_experiments:
            return 0.0
        total_time = sum(qe.latency for qe in self.query_experiments)
        return len(self.query_experiments) / total_time if total_time > 0 else 0.0
    
    def calculate_average_memory(self) -> float:
        """Calculate average memory usage per query"""
        if not self.query_experiments:
            return 0.0
        return sum(qe.memory_used for qe in self.query_experiments) / len(self.query_experiments)

class ExperimentTracker:
    """Class to track and store experimental measurements for multiple models"""
    
    def __init__(self):
        self.experiments = {}
        self.hardware_info = {}
        self._temp_metrics = {}  # Temporary storage for calculating aggregates
    
    def _get_hardware_info(self, device: str) -> Dict:
        """Gather hardware information only for the device used"""
        info = {}
        
        if device == "cpu":
            cpu_freq = psutil.cpu_freq()
            info["device_type"] = "cpu"
            info["hardware_specs"] = {
                "processor": platform.processor(),
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_frequency": f"{cpu_freq.max:.2f}MHz" if cpu_freq else "Unknown",
                "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
                "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f}GB"
            }
        elif device == "cuda" and torch.cuda.is_available():
            info["device_type"] = "gpu"
            info["hardware_specs"] = {
                "name": torch.cuda.get_device_name(0),
                "compute_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
                "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}GB",
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / (1024**3):.2f}GB",
                "memory_cached": f"{torch.cuda.memory_reserved(0) / (1024**3):.2f}GB"
            }
        
        return info
    
    def start_experiment(self, model_name: str, device: str, retrieval_model: str = None):
        """Initialize experiment tracking"""
        self.hardware_info = self._get_hardware_info(device)
        self.experiments[model_name] = {
            "device": device,  # Store device type directly
            "model_name": model_name,
            "retrieval_model": retrieval_model,
            "model_dimension": 0,
            "index_size_mb": 0.0,
            "model_size_mb": 0.0,
            "load_time": 0.0,
            "warmup_time": 0.0,
            "index_time": 0.0,
            "total_latency": 0.0,
            "total_memory": 0.0,
            "query_count": 0
        }
    
    def track_search(self, model_name: str):
        """Track search performance"""
        start_time = time.perf_counter()
        
        def end_tracking():
            total_time = time.perf_counter() - start_time
            
            self.experiments[model_name]['total_search_time'] = total_time 
            
        return end_tracking
    
    def _get_memory_usage(self) -> float:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def export_results(self, model_name: str, timestamp: str):
        """Export metrics to inference/measured_data/model_name/timestamp/experiment_tracked.json"""
        current_file = Path(__file__)
        inference_dir = current_file.parent
        
        base_dir = inference_dir / "measured_data" / model_name.lower() / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_name = f"{model_name.lower()}.experiment"
        output_path = base_dir / f"{experiment_name}_tracked.json"
        
        results_dict = {
            "hardware_info": self.hardware_info,
            "experiment": {
                "experiment_name": experiment_name,
                "device": self.experiments[model_name]["device"],
                "model_name": model_name,
                "retrieval_model": self.experiments[model_name].get("retrieval_model"),
                "model_dimension": self.experiments[model_name]["model_dimension"],
                "index_size_mb": self.experiments[model_name]["index_size_mb"],
                "model_size_mb": self.experiments[model_name]["model_size_mb"],
                "load_time": self.experiments[model_name]["load_time"],
                "warmup_time": self.experiments[model_name]["warmup_time"],
                "index_time": self.experiments[model_name]["index_time"],
                "average_latency": self.calculate_average_latency(model_name),
                "throughput": self.calculate_throughput(model_name),
                "average_memory": self.calculate_average_memory(model_name)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def calculate_average_latency(self, model_name: str) -> float:
        """Calculate average latency per query"""
        exp = self.experiments[model_name]
        return exp['total_search_time'] / exp['total queries']
    
    def calculate_throughput(self, model_name: str) -> float:
        """Calculate queries per second"""
        exp = self.experiments[model_name]
        return exp['total queries'] / exp['total_search_time']
    
    def calculate_average_memory(self, model_name: str) -> float:
        exp = self.experiments[model_name]
        return exp["total_memory"] / exp["query_count"] if exp["query_count"] > 0 else 0.0

# Make sure these are available for import
__all__ = ['ExperimentTracker', 'QueryExperiment', 'ModelExperiment']
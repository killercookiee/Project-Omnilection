# model_clustering.py
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class ModelClusterer:
    def __init__(self,
        model_registry_path: str = 'LLM_models/model_registry.json',
        n_clusters: int = 5,
        feature_config: List[Tuple[str, float]] = [
            ('parameters_count', 1.0),
            ('context_length', 1.0),
            ('benchmark.total_duration', 1.0),
            ('architecture', 0.2)
        ]
    ):
        """
        Initialize the model clusterer.
        
        Parameters:
        - model_registry_path: Path to the model registry JSON file
        - n_clusters: Number of clusters for k-means (default: 5)
        - feature_config: List of tuples [(feature_name, weight), ...]
                         Default: [('parameters_count', 1.0), ('context_length', 1.0), ('benchmark.total_duration', 1.0)]
                         Supports:
                           - Numeric features: 'context_length', 'benchmark.eval_rate'
                           - Text features: 'architecture', 'quantization'
                           - List features: 'capabilities', 'stop_tokens'
        """
        self.model_registry_path = model_registry_path
        self.n_clusters = n_clusters
        self.feature_config = feature_config
        
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.model_names: List[str] = []
        self.feature_matrix: np.ndarray = None
        self.normalized_features: np.ndarray = None
        self.scaler: MinMaxScaler = None
        self.kmeans: KMeans = None
        self.cluster_labels: np.ndarray = None
        
        # Feature encoding metadata
        self.feature_types: Dict[str, str] = {}  # 'numeric', 'text', or 'list'
        self.text_encoders: Dict[str, LabelEncoder] = {}  # For categorical features
        self.list_vocabularies: Dict[str, List[str]] = {}  # For list features
        self.feature_dimensions: Dict[str, int] = {}  # Number of columns per feature
        
        # Load and process
        self.load_model_registry()
        self._detect_feature_types()
        self.extract_features()
        self.fit_clusters()
    
    def load_model_registry(self) -> None:
        """Load the model registry from the JSON file."""
        try:
            with open(self.model_registry_path, 'r') as f:
                self.model_registry = json.load(f)
            self.model_names = list(self.model_registry.keys())
            print(f"Loaded {len(self.model_registry)} models from registry")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model registry not found at {self.model_registry_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing model registry: {e}")
    
    def extract_raw_value(self, model_data: Dict[str, Any], feature_name: str) -> Any:
        """
        Extract a feature value from model data using dot notation for nested keys.
        
        Parameters:
        - model_data: Dictionary containing model information
        - feature_name: Name of the feature to extract (supports dot notation for nested keys)
        
        Returns:
        - The raw value of the feature, or None if not found
        """
        keys = feature_name.split('.')
        value = model_data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return None
            else:
                return None
        
        return value
    
    def _detect_feature_types(self) -> None:
        """
        Automatically detect the type of each feature by sampling the first model.
        """
        if not self.model_names:
            raise ValueError("No models loaded")
        
        first_model = self.model_registry[self.model_names[0]]
        
        for feature_name, _ in self.feature_config:
            value = self.extract_raw_value(first_model, feature_name)
            
            if value is None:
                # Try to infer from other models
                for model_name in self.model_names[1:]:
                    value = self.extract_raw_value(self.model_registry[model_name], feature_name)
                    if value is not None:
                        break
            
            if value is None:
                print(f"Warning: Feature '{feature_name}' not found in any model, treating as numeric (will be 0)")
                self.feature_types[feature_name] = 'numeric'
            elif isinstance(value, (int, float)):
                self.feature_types[feature_name] = 'numeric'
            elif isinstance(value, str):
                self.feature_types[feature_name] = 'text'
            elif isinstance(value, list):
                self.feature_types[feature_name] = 'list'
            else:
                print(f"Warning: Unknown type for feature '{feature_name}': {type(value)}, treating as numeric")
                self.feature_types[feature_name] = 'numeric'
        
        print(f"\nDetected feature types:")
        for feature_name, feature_type in self.feature_types.items():
            print(f"  {feature_name}: {feature_type}")
    
    def _prepare_text_encoders(self) -> None:
        """
        Prepare label encoders for text features.
        Collect all unique values across all models.
        """
        for feature_name, feature_type in self.feature_types.items():
            if feature_type == 'text':
                # Collect all unique values
                unique_values = set()
                for model_name in self.model_names:
                    value = self.extract_raw_value(self.model_registry[model_name], feature_name)
                    if value is not None:
                        unique_values.add(str(value))
                
                # Create and fit label encoder
                encoder = LabelEncoder()
                encoder.fit(list(unique_values))
                self.text_encoders[feature_name] = encoder
                self.feature_dimensions[feature_name] = 1
                
                print(f"  Text feature '{feature_name}': {len(unique_values)} unique values")
    
    def _prepare_list_vocabularies(self) -> None:
        """
        Prepare vocabularies for list features.
        Collect all unique items across all models for multi-hot encoding.
        """
        for feature_name, feature_type in self.feature_types.items():
            if feature_type == 'list':
                # Collect all unique items
                all_items = []
                for model_name in self.model_names:
                    value = self.extract_raw_value(self.model_registry[model_name], feature_name)
                    if value is not None and isinstance(value, list):
                        all_items.extend([str(item) for item in value])
                
                # Create vocabulary
                unique_items = sorted(set(all_items))
                self.list_vocabularies[feature_name] = unique_items
                self.feature_dimensions[feature_name] = len(unique_items)
                
                print(f"  List feature '{feature_name}': {len(unique_items)} unique items - {unique_items}")
    
    def _encode_numeric_feature(self, value: Any) -> float:
        """Convert a value to numeric, handling None and invalid values."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _encode_text_feature(self, value: Any, feature_name: str) -> float:
        """Encode a text feature using label encoding."""
        if value is None:
            return 0.0
        
        encoder = self.text_encoders[feature_name]
        try:
            return float(encoder.transform([str(value)])[0])
        except ValueError:
            # Unknown category, return 0
            return 0.0
    
    def _encode_list_feature(self, value: Any, feature_name: str) -> np.ndarray:
        """Encode a list feature using multi-hot encoding."""
        vocab = self.list_vocabularies[feature_name]
        encoding = np.zeros(len(vocab))
        
        if value is not None and isinstance(value, list):
            for item in value:
                item_str = str(item)
                if item_str in vocab:
                    idx = vocab.index(item_str)
                    encoding[idx] = 1.0
        
        return encoding
    
    def extract_features(self) -> None:
        """Extract and encode features from all models in the registry."""
        # Prepare encoders
        print("\nPreparing feature encoders:")
        self._prepare_text_encoders()
        self._prepare_list_vocabularies()
        
        # Calculate total number of feature dimensions
        total_dims = 0
        for feature_name, _ in self.feature_config:
            feature_type = self.feature_types[feature_name]
            if feature_type == 'numeric' or feature_type == 'text':
                total_dims += 1
                self.feature_dimensions[feature_name] = 1
            elif feature_type == 'list':
                total_dims += self.feature_dimensions[feature_name]
        
        print(f"\nTotal feature dimensions: {total_dims}")
        
        n_models = len(self.model_names)
        self.feature_matrix = np.zeros((n_models, total_dims))
        
        # Extract features for each model
        for i, model_name in enumerate(self.model_names):
            model_data = self.model_registry[model_name]
            col_idx = 0
            
            for feature_name, weight in self.feature_config:
                feature_type = self.feature_types[feature_name]
                raw_value = self.extract_raw_value(model_data, feature_name)
                
                if feature_type == 'numeric':
                    self.feature_matrix[i, col_idx] = self._encode_numeric_feature(raw_value) * weight
                    col_idx += 1
                
                elif feature_type == 'text':
                    self.feature_matrix[i, col_idx] = self._encode_text_feature(raw_value, feature_name) * weight
                    col_idx += 1
                
                elif feature_type == 'list':
                    encoded = self._encode_list_feature(raw_value, feature_name) * weight
                    n_dims = len(encoded)
                    self.feature_matrix[i, col_idx:col_idx + n_dims] = encoded
                    col_idx += n_dims
        
        # Normalize using min-max scaling
        self.scaler = MinMaxScaler()
        self.normalized_features = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"\nExtracted features from {n_models} models into {total_dims}D feature space")
    
    def fit_clusters(self) -> None:
        """Fit k-means clustering on the normalized features."""
        if self.normalized_features is None:
            raise ValueError("Features not extracted. Call extract_features() first.")
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.normalized_features)
        
        print(f"\nFitted {self.n_clusters} clusters")
        
        # Print cluster distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            models_in_cluster = [self.model_names[i] for i in range(len(self.model_names)) 
                                if self.cluster_labels[i] == cluster_id]
            print(f"  Cluster {cluster_id}: {count} models - {models_in_cluster}")
    
    def get_model_cluster(self, model_name: str) -> int:
        """Get the cluster ID for a specific model."""
        if model_name not in self.model_names:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        idx = self.model_names.index(model_name)
        return int(self.cluster_labels[idx])
    
    def get_models_in_cluster(self, cluster_id: int) -> List[str]:
        """Get all models in a specific cluster."""
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}. Must be 0-{self.n_clusters-1}")
        
        return [self.model_names[i] for i in range(len(self.model_names)) 
                if self.cluster_labels[i] == cluster_id]
    
    def get_distance(self, model1: str, model2: str) -> float:
        """
        Calculate the normalized Euclidean distance between two models.
        
        Parameters:
        - model1: Name of the first model
        - model2: Name of the second model
        
        Returns:
        - Normalized distance
        """
        if model1 not in self.model_names or model2 not in self.model_names:
            return float('inf')  # Maximum distance if model not found
        
        idx1 = self.model_names.index(model1)
        idx2 = self.model_names.index(model2)
        
        # Euclidean distance in normalized feature space
        distance = np.linalg.norm(self.normalized_features[idx1] - self.normalized_features[idx2])
        
        return float(distance)
    
    def get_all_distances(self, model_name: str, exclude_models: List[str] = None) -> Dict[str, float]:
        """
        Get distances from a model to all other models.
        
        Parameters:
        - model_name: Name of the source model
        - exclude_models: List of model names to exclude
        
        Returns:
        - Dictionary mapping model names to their distances, sorted by distance
        """
        if model_name not in self.model_names:
            return {}
        
        exclude_models = exclude_models or []
        distances = {}
        
        for other_model in self.model_names:
            if other_model != model_name and other_model not in exclude_models:
                distances[other_model] = self.get_distance(model_name, other_model)
        
        return dict(sorted(distances.items(), key=lambda x: x[1]))
    
    def get_similar_models(self, model_name: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the k most similar models to a given model.
        
        Parameters:
        - model_name: Name of the source model
        - top_k: Number of similar models to return
        
        Returns:
        - List of tuples (model_name, distance) sorted by similarity
        """
        distances = self.get_all_distances(model_name)
        sorted_models = sorted(distances.items(), key=lambda x: x[1])
        return sorted_models[:top_k]


if __name__ == "__main__":
    # Example 1: Cluster by numeric features only
    print("=" * 80)
    print("Example 1: Clustering by numeric features")
    print("=" * 80)
    clusterer1 = ModelClusterer(
        model_registry_path='LLM_models/model_registry.json',
        n_clusters=3,
        feature_config=[
            ('parameters_count', 1.0),
            ('context_length', 0.5),
            ('benchmark.total_duration', 1.0)
        ]
    )
    
    # Show all clusters for Example 1
    print("\nAll clusters in Example 1:")
    for i in range(clusterer1.n_clusters):
        models = clusterer1.get_models_in_cluster(i)
        print(f"  Cluster {i} ({len(models)} models): {models}")
    
    # Get all distances to qwen3:0.6b
    print("\nDistances from 'qwen3:0.6b' to all other models (Example 1):")
    distances1 = clusterer1.get_all_distances('qwen3:0.6b')
    for model, dist in distances1.items():
        print(f"  {model}: {dist:.4f}")
    
    # Example 2: Cluster by architecture feature
    print("\n" + "=" * 80)
    print("Example 2: Clustering by architecture feature")
    print("=" * 80)
    clusterer2 = ModelClusterer(
        model_registry_path='LLM_models/model_registry.json',
        n_clusters=4,
        feature_config=[
            ('architecture', 2.0),  # Text feature
        ]
    )
    
    # Show all clusters for Example 2
    print("\nAll clusters in Example 2:")
    for i in range(clusterer2.n_clusters):
        models = clusterer2.get_models_in_cluster(i)
        print(f"  Cluster {i} ({len(models)} models): {models}")
    
    # Get all distances to qwen3:0.6b
    print("\nDistances from 'qwen3:0.6b' to all other models (Example 2):")
    distances2 = clusterer2.get_all_distances('qwen3:0.6b')
    for model, dist in distances2.items():
        print(f"  {model}: {dist:.4f}")
    
    # Example 3: Cluster by mixed features (numeric + text + list)
    print("\n" + "=" * 80)
    print("Example 3: Clustering by mixed features")
    print("=" * 80)
    clusterer3 = ModelClusterer(
        model_registry_path='LLM_models/model_registry.json',
        n_clusters=5,
        feature_config=[
            ('parameters_count', 1.0),
            ('architecture', 2.0),      # Text feature
            ('capabilities', 1.5),       # List feature
            ('quantization', 0.5),       # Text feature
            ('context_length', 0.8)
        ]
    )
    
    # Show all clusters for Example 3
    print("\nAll clusters in Example 3:")
    for i in range(clusterer3.n_clusters):
        models = clusterer3.get_models_in_cluster(i)
        print(f"  Cluster {i} ({len(models)} models): {models}")
    
    # Get all distances to qwen3:0.6b
    print("\nDistances from 'qwen3:0.6b' to all other models (Example 3):")
    distances3 = clusterer3.get_all_distances('qwen3:0.6b')
    for model, dist in distances3.items():
        print(f"  {model}: {dist:.4f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("Summary: Top 3 most similar models to 'qwen3:0.6b' in each example")
    print("=" * 80)
    
    print("\nExample 1 (Numeric features):")
    similar1 = clusterer1.get_similar_models('qwen3:0.6b', top_k=3)
    for model, dist in similar1:
        print(f"  {model}: distance = {dist:.4f}")
    
    print("\nExample 2 (Architecture feature):")
    similar2 = clusterer2.get_similar_models('qwen3:0.6b', top_k=3)
    for model, dist in similar2:
        print(f"  {model}: distance = {dist:.4f}")
    
    print("\nExample 3 (Mixed features):")
    similar3 = clusterer3.get_similar_models('qwen3:0.6b', top_k=3)
    for model, dist in similar3:
        print(f"  {model}: distance = {dist:.4f}")
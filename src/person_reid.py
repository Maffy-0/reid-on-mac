"""
Person re-identification using FastReID features and similarity matching.
"""
import pickle
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .logger import logger


class PersonReID:
    """Person re-identification using feature vectors and similarity matching."""
    
    def __init__(self):
        self.device = config.DEVICE
        self.similarity_threshold = config.REID_SIMILARITY_THRESHOLD
        self.feature_dim = config.REID_FEATURE_DIM
        self.templates_dir = config.TEMPLATES_DIR
        
        # Person database: {person_id: List[feature_vector]}
        self.person_database: Dict[str, List[np.ndarray]] = {}
        self.load_person_database()
        
        # Simple feature extractor (ResNet-like CNN)
        self.feature_extractor = self._create_feature_extractor()
        
    def _create_feature_extractor(self):
        """Create a simple CNN feature extractor."""
        class SimpleFeatureExtractor(torch.nn.Module):
            def __init__(self, feature_dim=2048):
                super().__init__()
                # Simple CNN layers
                self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(128, 256, 3, padding=1)
                self.conv4 = torch.nn.Conv2d(256, 512, 3, padding=1)
                
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(512, feature_dim)
                self.dropout = torch.nn.Dropout(0.5)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv3(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv4(x))
                x = self.pool(x)
                x = torch.flatten(x, 1)
                x = self.dropout(x)
                x = self.fc(x)
                return F.normalize(x, p=2, dim=1)
        
        model = SimpleFeatureExtractor(self.feature_dim)
        model.to(self.device)
        model.eval()
        return model
        
    def extract_features(self, person_crop: np.ndarray) -> np.ndarray:
        """Extract feature vector from person crop."""
        try:
            # Preprocess image
            if len(person_crop.shape) == 3 and person_crop.shape[2] == 3:
                # Resize to standard size
                crop_resized = cv2.resize(person_crop, (128, 256))
                
                # Convert to tensor and normalize
                crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float()
                crop_tensor = crop_tensor / 255.0
                crop_tensor = crop_tensor.unsqueeze(0).to(self.device)
                
                # Extract features
                self.feature_extractor.eval()  # Ensure eval mode
                with torch.no_grad():
                    features = self.feature_extractor(crop_tensor)
                    features = features.cpu().numpy().flatten()
                
                return features
            else:
                logger.log_warning("Invalid person crop format")
                return np.zeros(self.feature_dim)
                
        except Exception as e:
            logger.log_error("Feature extraction failed", e)
            return np.zeros(self.feature_dim)
    
    def identify_person(self, person_crop: np.ndarray) -> Tuple[str, float]:
        """Identify person from crop image."""
        if len(self.person_database) == 0:
            return "Unknown", 0.0
            
        # Extract features from crop
        query_features = self.extract_features(person_crop)
        if np.all(query_features == 0):
            return "Unknown", 0.0
        
        # Compare with database
        best_match_id = "Unknown"
        best_similarity = 0.0
        
        for person_id, template_features_list in self.person_database.items():
            # Calculate similarity with all templates for this person
            similarities = []
            
            for template_features in template_features_list:
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    template_features.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
            
            # Use the maximum similarity among all templates
            max_similarity = max(similarities) if similarities else 0.0
            
            # Alternative: Use average similarity
            # avg_similarity = np.mean(similarities) if similarities else 0.0
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match_id = person_id
        
        # Apply threshold
        if best_similarity < self.similarity_threshold:
            best_match_id = "Unknown"
            
        return best_match_id, best_similarity
    
    def add_person_template(self, person_id: str, person_crop: np.ndarray, max_templates: int = 5):
        """
        Add a new person template to the database.
        
        Args:
            person_id: Identifier for the person
            person_crop: Cropped image of the person
            max_templates: Maximum number of templates to keep per person
        """
        features = self.extract_features(person_crop)
        if not np.all(features == 0):
            # Initialize list if person is new
            if person_id not in self.person_database:
                self.person_database[person_id] = []
            
            # Add new template
            self.person_database[person_id].append(features)
            
            # Limit number of templates per person
            if len(self.person_database[person_id]) > max_templates:
                # Remove oldest template (FIFO)
                self.person_database[person_id] = self.person_database[person_id][-max_templates:]
            
            self.save_person_database()
            logger.log_info(f"Added template for {person_id} (total: {len(self.person_database[person_id])})")
        else:
            logger.log_warning(f"Failed to extract features for {person_id}")
    
    def load_person_database(self):
        """Load person database from file."""
        db_file = self.templates_dir / "person_database.pkl"
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    loaded_data = pickle.load(f)
                    
                # Handle backward compatibility
                if loaded_data and isinstance(list(loaded_data.values())[0], np.ndarray):
                    # Old format: single template per person
                    logger.log_info("Converting old database format to new multi-template format")
                    self.person_database = {person_id: [features] 
                                          for person_id, features in loaded_data.items()}
                else:
                    # New format: multiple templates per person
                    self.person_database = loaded_data
                    
                total_templates = sum(len(templates) for templates in self.person_database.values())
                logger.log_info(f"Loaded {len(self.person_database)} persons with {total_templates} total templates")
                
            except Exception as e:
                logger.log_error("Failed to load person database", e)
                self.person_database = {}
        else:
            # Create sample templates
            self._create_sample_templates()
    
    def save_person_database(self):
        """Save person database to file."""
        db_file = self.templates_dir / "person_database.pkl"
        try:
            with open(db_file, 'wb') as f:
                pickle.dump(self.person_database, f)
            logger.log_info("Person database saved")
        except Exception as e:
            logger.log_error("Failed to save person database", e)
    
    def _create_sample_templates(self):
        """Create sample person templates."""
        sample_persons = ["person_1", "person_2", "person_3"]
        
        for person_id in sample_persons:
            # Create multiple random feature vectors as placeholders for each person
            self.person_database[person_id] = []
            
            # Add 2-3 templates per person for better matching
            num_templates = np.random.randint(2, 4)
            for i in range(num_templates):
                features = np.random.rand(self.feature_dim)
                features = features / np.linalg.norm(features)  # Normalize
                self.person_database[person_id].append(features)
            
        self.save_person_database()
        total_templates = sum(len(templates) for templates in self.person_database.values())
        logger.log_info(f"Created sample person templates: {total_templates} total")
    
    def get_registered_persons(self) -> List[str]:
        """Get list of registered person IDs."""
        return list(self.person_database.keys())
    
    def list_person_templates(self) -> Dict[str, int]:
        """List all person templates with their counts."""
        template_counts = {}
        for person_id, templates in self.person_database.items():
            if isinstance(templates, list):
                template_counts[person_id] = len(templates)
            else:
                # Legacy single template format
                template_counts[person_id] = 1
        return template_counts
    
    def get_person_template_count(self, person_id: str) -> int:
        """Get the number of templates for a specific person."""
        if person_id not in self.person_database:
            return 0
        
        templates = self.person_database[person_id]
        if isinstance(templates, list):
            return len(templates)
        else:
            # Legacy single template format
            return 1
    
    def remove_person_all_templates(self, person_id: str) -> bool:
        """Remove all templates for a person."""
        if person_id in self.person_database:
            del self.person_database[person_id]
            self.save_person_database()
            logger.log_info(f"Removed all templates for {person_id}")
            return True
        return False
    
    def get_similarity_statistics(self, person_crop: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Get detailed similarity statistics for a person crop against all templates.
        
        Returns:
            Dictionary with person_id -> {'max': float, 'min': float, 'avg': float, 'count': int}
        """
        if len(self.person_database) == 0:
            return {}
            
        query_features = self.extract_features(person_crop)
        if np.all(query_features == 0):
            return {}
        
        stats = {}
        
        for person_id, template_features_list in self.person_database.items():
            similarities = []
            
            for template_features in template_features_list:
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    template_features.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
            
            if similarities:
                stats[person_id] = {
                    'max': max(similarities),
                    'min': min(similarities),
                    'avg': np.mean(similarities),
                    'count': len(similarities)
                }
        
        return stats

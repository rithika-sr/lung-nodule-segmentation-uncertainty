"""
Data Preprocessing Module for LUNA16 Lung Nodule Segmentation
Extracts patches around nodules and prepares data for training
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import pickle
from tqdm import tqdm

class LUNA16Preprocessor:
    """
    Preprocessor for LUNA16 dataset
    Extracts nodule patches and creates training data
    """
    
    def __init__(self, 
                 raw_data_dir='data/raw/',
                 processed_data_dir='data/processed/',
                 patch_size=64):
        """
        Initialize preprocessor
        
        Args:
            raw_data_dir: Directory containing raw LUNA16 data
            processed_data_dir: Directory to save processed data
            patch_size: Size of extracted patches (cubic)
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.patch_size = patch_size
        
        # Create processed directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        self.annotations_df = pd.read_csv(self.raw_data_dir / 'annotations.csv')
        self.candidates_df = pd.read_csv(self.raw_data_dir / 'candidates.csv')
        
        print(f"✅ Loaded {len(self.annotations_df)} annotations")
        print(f"✅ Loaded {len(self.candidates_df)} candidates")
        
    def load_ct_scan(self, series_uid):
        """
        Load a CT scan given its series UID
        
        Args:
            series_uid: Unique identifier for the CT scan
            
        Returns:
            ct_array: 3D numpy array of the CT scan
            origin: Origin coordinates
            spacing: Voxel spacing
        """
        # Find the .mhd file
        scan_dir = self.raw_data_dir / 'seg-lungs-LUNA16' / 'seg-lungs-LUNA16'
        mhd_file = scan_dir / f"{series_uid}.mhd"
        
        if not mhd_file.exists():
            raise FileNotFoundError(f"Scan not found: {mhd_file}")
        
        # Load using SimpleITK
        ct_scan = sitk.ReadImage(str(mhd_file))
        ct_array = sitk.GetArrayFromImage(ct_scan)
        origin = np.array(ct_scan.GetOrigin())
        spacing = np.array(ct_scan.GetSpacing())
        
        return ct_array, origin, spacing
    
    def world_to_voxel(self, world_coords, origin, spacing):
        """
        Convert world coordinates (mm) to voxel coordinates
        
        Args:
            world_coords: Coordinates in world space (x, y, z)
            origin: Origin of the CT scan
            spacing: Voxel spacing
            
        Returns:
            voxel_coords: Coordinates in voxel space (z, y, x)
        """
        voxel_coords = np.abs((world_coords - origin) / spacing)
        # Note: SimpleITK uses (x, y, z) but numpy uses (z, y, x)
        return np.array([voxel_coords[2], voxel_coords[1], voxel_coords[0]]).astype(int)
    
    def extract_patch(self, ct_array, center_voxel, patch_size):
        """
        Extract a 3D patch from CT scan centered at given voxel
        
        Args:
            ct_array: 3D CT scan array
            center_voxel: Center coordinates (z, y, x)
            patch_size: Size of the patch to extract
            
        Returns:
            patch: 3D patch of size (patch_size, patch_size, patch_size)
        """
        z, y, x = center_voxel
        half_size = patch_size // 2
        
        # Calculate patch boundaries
        z_start = max(0, z - half_size)
        z_end = min(ct_array.shape[0], z + half_size)
        y_start = max(0, y - half_size)
        y_end = min(ct_array.shape[1], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(ct_array.shape[2], x + half_size)
        
        # Extract patch
        patch = ct_array[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Pad if necessary (if patch is near boundary)
        if patch.shape != (patch_size, patch_size, patch_size):
            padded_patch = np.zeros((patch_size, patch_size, patch_size))
            
            # Calculate padding offsets
            z_offset = (patch_size - patch.shape[0]) // 2
            y_offset = (patch_size - patch.shape[1]) // 2
            x_offset = (patch_size - patch.shape[2]) // 2
            
            # Place patch in center
            padded_patch[
                z_offset:z_offset + patch.shape[0],
                y_offset:y_offset + patch.shape[1],
                x_offset:x_offset + patch.shape[2]
            ] = patch
            
            return padded_patch
        
        return patch
    
    def normalize_patch(self, patch):
        """
        Normalize patch values to [0, 1] range
        
        Args:
            patch: 3D patch
            
        Returns:
            normalized_patch: Normalized patch
        """
        # For binary lung masks, just normalize to [0, 1]
        if patch.max() > 0:
            return patch / patch.max()
        return patch
    
    def create_mask_from_diameter(self, diameter_mm, spacing, patch_size):
        """
        Create a spherical mask given nodule diameter
        
        Args:
            diameter_mm: Nodule diameter in millimeters
            spacing: Voxel spacing
            patch_size: Size of the patch
            
        Returns:
            mask: Binary mask with nodule shape
        """
        # Convert diameter to voxels
        radius_voxels = (diameter_mm / 2) / np.mean(spacing)
        
        # Create coordinate grids
        center = patch_size // 2
        z, y, x = np.ogrid[:patch_size, :patch_size, :patch_size]
        
        # Create spherical mask
        distance = np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2)
        mask = (distance <= radius_voxels).astype(np.float32)
        
        return mask
    
    def process_dataset(self, max_samples=100, negative_ratio=2):
        """
        Process the entire dataset and save patches
        
        Args:
            max_samples: Maximum number of positive samples to process
            negative_ratio: Number of negative samples per positive sample
            
        Returns:
            None (saves processed data to disk)
        """
        print(f"\n{'='*60}")
        print("STARTING DATA PREPROCESSING")
        print(f"{'='*60}")
        print(f"Patch size: {self.patch_size}x{self.patch_size}x{self.patch_size}")
        print(f"Max positive samples: {max_samples}")
        print(f"Negative ratio: {negative_ratio}")
        
        positive_patches = []
        positive_masks = []
        negative_patches = []
        
        # Get unique series UIDs with nodules
        unique_series = self.annotations_df['seriesuid'].unique()[:max_samples]
        
        print(f"\nProcessing {len(unique_series)} scans...")
        
        for series_uid in tqdm(unique_series, desc="Processing scans"):
            try:
                # Load CT scan
                ct_array, origin, spacing = self.load_ct_scan(series_uid)
                
                # Get all nodules for this scan
                scan_nodules = self.annotations_df[
                    self.annotations_df['seriesuid'] == series_uid
                ]
                
                for idx, nodule in scan_nodules.iterrows():
                    # Extract positive sample (nodule)
                    world_coords = np.array([
                        nodule['coordX'],
                        nodule['coordY'],
                        nodule['coordZ']
                    ])
                    
                    voxel_coords = self.world_to_voxel(world_coords, origin, spacing)
                    
                    # Extract nodule patch
                    patch = self.extract_patch(ct_array, voxel_coords, self.patch_size)
                    patch = self.normalize_patch(patch)
                    
                    # Create segmentation mask
                    mask = self.create_mask_from_diameter(
                        nodule['diameter_mm'], 
                        spacing, 
                        self.patch_size
                    )
                    
                    positive_patches.append(patch)
                    positive_masks.append(mask)
                    
                    # Extract negative samples (random locations without nodules)
                    for _ in range(negative_ratio):
                        # Random coordinates within scan bounds
                        random_z = np.random.randint(self.patch_size//2, 
                                                    ct_array.shape[0] - self.patch_size//2)
                        random_y = np.random.randint(self.patch_size//2, 
                                                    ct_array.shape[1] - self.patch_size//2)
                        random_x = np.random.randint(self.patch_size//2, 
                                                    ct_array.shape[2] - self.patch_size//2)
                        
                        random_voxel = np.array([random_z, random_y, random_x])
                        
                        neg_patch = self.extract_patch(ct_array, random_voxel, self.patch_size)
                        neg_patch = self.normalize_patch(neg_patch)
                        
                        negative_patches.append(neg_patch)
                
            except Exception as e:
                print(f"\n⚠️  Error processing {series_uid}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        positive_patches = np.array(positive_patches)
        positive_masks = np.array(positive_masks)
        negative_patches = np.array(negative_patches)
        
        # Create labels
        positive_labels = np.ones(len(positive_patches))
        negative_labels = np.zeros(len(negative_patches))
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f" Positive samples: {len(positive_patches)}")
        print(f" Negative samples: {len(negative_patches)}")
        print(f" Positive patch shape: {positive_patches.shape}")
        print(f" Mask shape: {positive_masks.shape}")
        
        # Save processed data
        print(f"\n Saving processed data to {self.processed_data_dir}...")
        
        np.save(self.processed_data_dir / 'positive_patches.npy', positive_patches)
        np.save(self.processed_data_dir / 'positive_masks.npy', positive_masks)
        np.save(self.processed_data_dir / 'negative_patches.npy', negative_patches)
        np.save(self.processed_data_dir / 'positive_labels.npy', positive_labels)
        np.save(self.processed_data_dir / 'negative_labels.npy', negative_labels)
        
        # Save metadata
        metadata = {
            'patch_size': self.patch_size,
            'num_positive': len(positive_patches),
            'num_negative': len(negative_patches),
            'negative_ratio': negative_ratio
        }
        
        with open(self.processed_data_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(" All data saved successfully!")
        
        return metadata


# Utility function for easy usage
def preprocess_luna16(max_samples=100, patch_size=64, negative_ratio=2):
    """
    Convenience function to run preprocessing
    
    Args:
        max_samples: Maximum number of positive samples
        patch_size: Size of patches to extract
        negative_ratio: Negative samples per positive sample
    """
    preprocessor = LUNA16Preprocessor(
        raw_data_dir='../data/raw/',
        processed_data_dir='../data/processed/',
        patch_size=patch_size
    )
    
    return preprocessor.process_dataset(
        max_samples=max_samples,
        negative_ratio=negative_ratio
    )


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing LUNA16 Preprocessor...")
    metadata = preprocess_luna16(max_samples=50, patch_size=64, negative_ratio=2)
    print(f"\n Preprocessing complete! Processed {metadata['num_positive']} positive samples.")
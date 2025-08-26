"""
Image operations for Cellpose GUI
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer, Michael Rariden and Marius Pachitariu.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import ndimage


class ImageOperation(ABC):
    """Base class for image operations"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the operation as it appears in the GUI"""
        pass

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply the image operation to the image

        Args:
            image: Input image array, can be 2D or 3D (with channels)
            **kwargs: Additional parameters for the operation

        Returns:
            Processed image array with same shape as input
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """Return dict of parameters needed for this operation

        Returns:
            dict: Parameter names and default values
        """
        return {}


class SobelEdgeDetection(ImageOperation):
    """Sobel edge detection operation"""

    @property
    def name(self) -> str:
        return "Sobel Edge Detection"

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Apply Sobel edge detection to find edges in the image"""
        # Handle different image dimensions
        if image.ndim == 3:  # Multi-channel image (H, W, C)
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[-1]):
                channel = image[..., i].astype(np.float32)
                sx = ndimage.sobel(channel, axis=0)
                sy = ndimage.sobel(channel, axis=1)
                magnitude = np.sqrt(sx**2 + sy**2)
                # Normalize to 0-255 range for display
                if magnitude.max() > 0:
                    magnitude = (magnitude / magnitude.max() * 255).astype(np.float32)
                result[..., i] = magnitude
            return result
        else:  # Single channel image (H, W)
            image_float = image.astype(np.float32)
            sx = ndimage.sobel(image_float, axis=0)
            sy = ndimage.sobel(image_float, axis=1)
            magnitude = np.sqrt(sx**2 + sy**2)
            # Normalize to 0-255 range for display
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max() * 255).astype(np.float32)
            return magnitude


class ImageOperationRegistry:
    """Registry for all available image operations"""

    def __init__(self) -> None:
        self.operations: Dict[str, ImageOperation] = {}
        self._register_default_operations()

    def _register_default_operations(self) -> None:
        """Register built-in image operations"""
        self.register(SobelEdgeDetection())

    def register(self, operation: ImageOperation) -> None:
        """Register a new image operation

        Args:
            operation: Instance of ImageOperation
        """
        if not isinstance(operation, ImageOperation):
            raise ValueError("Operation must be an instance of ImageOperation")
        self.operations[operation.name] = operation

    def get_operation_names(self) -> List[str]:
        """Get list of all registered operation names"""
        return list(self.operations.keys())

    def get_operation(self, name: str) -> Optional[ImageOperation]:
        """Get operation instance by name

        Args:
            name: Name of the operation

        Returns:
            ImageOperation instance or None if not found
        """
        return self.operations.get(name)


# Global registry instance
image_operation_registry = ImageOperationRegistry()

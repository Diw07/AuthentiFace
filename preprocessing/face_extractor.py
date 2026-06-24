"""
Face extraction module using RetinaFace (insightface).
Detects faces in images, aligns using landmarks, and crops to target size.
"""

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from utils.config import (
    FACE_DET_SIZE,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_MARGIN,
    FACE_OUTPUT_SIZE,
)


class FaceExtractor:
    """
    Extracts faces from images using RetinaFace.

    Usage:
        extractor = FaceExtractor()
        faces = extractor.extract_faces(image)
        # faces is a list of (cropped_face_array, bbox, confidence)
    """

    def __init__(self, det_size=FACE_DET_SIZE, ctx_id=-1):
        """
        Initialize RetinaFace detector.

        Args:
            det_size: Detection resolution (width, height). Higher = more accurate but slower.
            ctx_id: GPU id (0, 1, ...) or -1 for CPU.
        """
        self.face_app = FaceAnalysis(
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.margin = FACE_MARGIN
        self.output_size = FACE_OUTPUT_SIZE
        self.confidence_threshold = FACE_CONFIDENCE_THRESHOLD

    def extract_faces(self, image, max_faces=1):
        """
        Detect and extract faces from an image.

        Args:
            image: numpy array (BGR, as from cv2.imread) or PIL Image.
            max_faces: Maximum number of faces to return (sorted by area, largest first).

        Returns:
            List of tuples: (face_crop, bbox, confidence)
            - face_crop: numpy array (RGB), resized to output_size x output_size
            - bbox: [x1, y1, x2, y2]
            - confidence: detection confidence score
        """
        # Convert PIL to numpy BGR if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Ensure image is valid
        if image is None or image.size == 0:
            return []

        # Detect faces
        faces = self.face_app.get(image)

        if not faces:
            return []

        # Filter by confidence
        faces = [f for f in faces if f.det_score >= self.confidence_threshold]

        if not faces:
            return []

        # Sort by face area (largest first)
        faces = sorted(faces, key=lambda f: self._face_area(f.bbox), reverse=True)

        # Limit to max_faces
        faces = faces[:max_faces]

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            confidence = float(face.det_score)

            # Crop face with margin
            face_crop = self._crop_face_with_margin(image, bbox)

            if face_crop is not None:
                # Convert BGR to RGB
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                results.append((face_crop, bbox.tolist(), confidence))

        return results

    def extract_single_face(self, image):
        """
        Extract the largest face from an image.

        Args:
            image: numpy array (BGR) or PIL Image.

        Returns:
            Tuple (face_crop, bbox, confidence) or None if no face found.
            - face_crop: numpy array (RGB), output_size x output_size x 3
        """
        faces = self.extract_faces(image, max_faces=1)
        return faces[0] if faces else None

    def _crop_face_with_margin(self, image, bbox):
        """
        Crop face region from image with margin, then resize to output_size.

        Args:
            image: numpy array (BGR)
            bbox: [x1, y1, x2, y2]

        Returns:
            Cropped and resized face (numpy array BGR), or None if crop fails.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        # Add margin
        margin_x = int((x2 - x1) * self.margin / 100)
        margin_y = int((y2 - y1) * self.margin / 100)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        # Crop
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # Resize to output size
        face_crop = cv2.resize(
            face_crop,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )

        return face_crop

    @staticmethod
    def _face_area(bbox):
        """Calculate area of a bounding box [x1, y1, x2, y2]."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

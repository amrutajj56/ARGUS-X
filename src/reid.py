
"""ARGUS-X general-purpose appearance feature extraction using DINOv2."""

import cv2
import numpy as np
import torch


class AppearanceExtractor:
    """Extract general-purpose visual embeddings using DINOv2."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

        # Load the pretrained lightweight DINOv2 model.
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
        )

        self.model.to(self.device)
        self.model.eval()

        self.input_size = 224

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

    def extract(self, frame, bbox):
        """Extract a normalized 384-dimensional appearance embedding."""

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # OpenCV uses BGR; DINOv2 expects RGB.
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        crop = cv2.resize(
            crop,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )

        image = torch.from_numpy(crop.copy())
        image = image.permute(2, 0, 1).float() / 255.0
        image = image.unsqueeze(0).to(self.device)

        image = (image - self.mean) / self.std

        with torch.inference_mode():
            output = self.model.forward_features(image)
            features = output["x_norm_clstoken"]

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=1,
            )

        return features.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def similarity(feature_a, feature_b):
        """Calculate cosine similarity between two embeddings."""

        if feature_a is None or feature_b is None:
            return 0.0

        if feature_a.shape != feature_b.shape:
            return 0.0

        return float(np.dot(feature_a, feature_b))

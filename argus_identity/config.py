from dataclasses import dataclass

@dataclass
class IdentityConfig:
    match_threshold: float = 0.65
    ambiguity_margin: float = 0.045
    max_lost_frames: int = 900
    gallery_size: int = 20
    min_box_area: int = 900
    min_crop_width: int = 16
    min_crop_height: int = 32
    min_detection_confidence: float = 0.10
    require_class_match: bool = True
    use_spatial_gate: bool = False
    max_center_distance: float = 0.35
    descriptor_bins: tuple = (16, 8, 8)

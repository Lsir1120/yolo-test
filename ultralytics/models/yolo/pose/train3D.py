from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER


class Detection3DTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending DetectionTrainer for 3D object detection using depth information.

    This trainer handles RGB-D input data and outputs 3D bounding box parameters
    (x, y, z, width, height, length, rotation).

    Attributes:
        args (dict): Configuration arguments with 'channels'=4 for RGB-D input
        loss_names (tuple): Names of 3D-specific loss components

    Methods:
        get_model: Creates model with 4-channel input support
        get_validator: Returns 3D validation handler
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "detect3d"
        overrides["channels"] = 4  # RGB + Depth
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> DetectionModel:
        """Get detection model with 4-channel input support."""
        model = DetectionModel(
            cfg, nc=self.data["nc"], ch=self.args.channels, verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return 3D detection validator."""
        self.loss_names = ("box_loss", "depth_loss", "dim_loss", "rot_loss", "cls_loss")
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_dataset(self) -> dict[str, Any]:
        """Ensure dataset includes depth information."""
        data = super().get_dataset()
        if self.args.channels != 4:
            raise ValueError("3D detection requires 4-channel input (RGB-D)")
        return data
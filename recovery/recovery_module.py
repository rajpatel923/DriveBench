"""
FrameRecoveryModule — differentiable nn.Module wrapping RIFE interpolation.

Phase 1 (DriveBench evaluation): used as a frozen preprocessing step.
  from recovery.recovery_module import FrameRecoveryModule
  module = FrameRecoveryModule.from_pretrained("recovery/rife/train_log")
  recovered = module(prev_frame, next_frame, t=0.5)

Phase 3 (UniDriveVLA fine-tuning): plug in before the vision encoder and
train end-to-end. The RIFE IFNet backbone is fully differentiable, so
gradients flow back through the recovered frame into UniDriveVLA's loss.

Gradient check:
  module = FrameRecoveryModule.from_pretrained("recovery/rife/train_log")
  prev = torch.rand(1, 3, 224, 224, requires_grad=True)
  nxt  = torch.rand(1, 3, 224, 224, requires_grad=True)
  out  = module(prev, nxt)
  out.mean().backward()
  assert prev.grad is not None   # gradients flow through RIFE

UniDriveVLA integration (pseudocode):
  # In UniDriveVLA forward(), before scene_perception_expert():
  if self.recovery_module is not None:
      camera_frames = self.recovery_module.recover_batch(
          camera_frames, prev_frames, next_frames, missing_mask
      )
  vision_tokens = self.scene_perception_expert(camera_frames)
"""

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image


class FrameRecoveryModule(nn.Module):
    """
    Wraps the RIFE IFNet as a differentiable frame-recovery pre-layer.

    forward() takes bracketing (prev, next) frame tensors and returns the
    interpolated frame at timestep t. recover_batch() handles per-frame
    missing masks for batched use inside UniDriveVLA.
    """

    def __init__(self, rife_net: nn.Module):
        """
        rife_net: the IFNet backbone from RIFE (already on device, eval mode).
                  Obtain via FrameRecoveryModule.from_pretrained().
        """
        super().__init__()
        self.net = rife_net

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, weights_dir: str, device: Optional[torch.device] = None) -> "FrameRecoveryModule":
        """
        Load RIFE HDv3 weights and return a FrameRecoveryModule.

        weights_dir: path to RIFE's train_log/ (e.g. "recovery/rife/train_log")
        Requires recovery/rife/ to be a clone of hzwer/ECCV2022-RIFE.
        """
        rife_repo = os.path.join(os.path.dirname(__file__), "rife")
        if not os.path.isdir(rife_repo):
            raise RuntimeError(
                f"RIFE repo not found at {rife_repo}.\n"
                "  git clone https://github.com/hzwer/ECCV2022-RIFE recovery/rife\n"
                "  # then download HDv3 weights into recovery/rife/train_log/"
            )
        if rife_repo not in sys.path:
            sys.path.insert(0, rife_repo)

        from model.RIFE_HDv3 import Model  # noqa: PLC0415

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rife_model = Model()
        rife_model.load_model(weights_dir, -1)
        rife_model.eval()
        rife_model.device()

        return cls(rife_model.net).to(device)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(
        self,
        prev_frame: torch.Tensor,
        next_frame: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """
        Interpolate between two frames at fractional timestep t.

        Args
        ----
        prev_frame : (B, 3, H, W) float tensor in [0, 1]
        next_frame : (B, 3, H, W) float tensor in [0, 1]
        t          : interpolation timestep, 0 = prev_frame, 1 = next_frame

        Returns
        -------
        (B, 3, H, W) float tensor in [0, 1]
        """
        imgs = torch.cat([prev_frame, next_frame], dim=1)   # (B, 6, H, W)
        scale_list = [8, 4, 2, 1]
        # RIFE IFNet forward returns (flow_list, mask_list, merged_list)
        _, _, merged = self.net(imgs, timestep=t, scale_list=scale_list)
        return merged[-1].clamp(0, 1)

    # ------------------------------------------------------------------
    # Batched recovery with missing-frame mask (for UniDriveVLA)
    # ------------------------------------------------------------------

    def recover_batch(
        self,
        frames: torch.Tensor,
        prev_frames: torch.Tensor,
        next_frames: torch.Tensor,
        missing_mask: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """
        Replace missing frames in a batch with RIFE interpolations.

        Args
        ----
        frames       : (B, 3, H, W) — current frames (some may be blank/corrupted)
        prev_frames  : (B, 3, H, W) — frames at t-1
        next_frames  : (B, 3, H, W) — frames at t+1
        missing_mask : (B,) bool tensor — True where frame is missing/corrupted
        t            : interpolation timestep

        Returns
        -------
        (B, 3, H, W) with missing positions replaced by recovered frames
        """
        if not missing_mask.any():
            return frames

        recovered = frames.clone()
        idx = missing_mask.nonzero(as_tuple=True)[0]
        interpolated = self.forward(prev_frames[idx], next_frames[idx], t=t)
        recovered[idx] = interpolated
        return recovered

    # ------------------------------------------------------------------
    # PIL convenience helpers (for offline dataset building)
    # ------------------------------------------------------------------

    def recover_pil(
        self,
        prev_img: Image.Image,
        next_img: Image.Image,
        t: float = 0.5,
    ) -> Image.Image:
        """Run recovery on PIL images; returns a PIL Image. No gradients."""
        device = next(self.net.parameters()).device
        I0 = TF.to_tensor(prev_img).unsqueeze(0).to(device)
        I1 = TF.to_tensor(next_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.forward(I0, I1, t=t)
        return TF.to_pil_image(out.squeeze(0).cpu())

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------

    def freeze(self):
        """Freeze all parameters (Phase 1 evaluation mode)."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze all parameters (Phase 3 joint fine-tuning)."""
        for p in self.parameters():
            p.requires_grad_(True)

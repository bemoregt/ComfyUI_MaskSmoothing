import numpy as np
import torch
import cv2


class MaskFourierSmoothing:
    """
    ComfyUI Custom Node: Mask Fourier Smoothing
    Uses Fourier Descriptors to smooth mask contours.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "num_descriptors": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 512,
                    "step": 4,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("smoothed_mask",)
    FUNCTION = "smooth_mask"
    CATEGORY = "mask/processing"

    def smooth_mask(self, mask: torch.Tensor, num_descriptors: int):
        # mask shape: (batch, H, W) or (H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        batch_size = mask.shape[0]
        results = []

        for i in range(batch_size):
            mask_np = mask[i].cpu().numpy()
            smoothed = self._apply_fourier_smoothing(mask_np, num_descriptors)
            results.append(smoothed)

        out = np.stack(results, axis=0)  # (batch, H, W)
        return (torch.from_numpy(out).float(),)

    def _apply_fourier_smoothing(self, mask_np: np.ndarray, num_descriptors: int) -> np.ndarray:
        H, W = mask_np.shape

        # Convert to uint8 binary mask
        binary = (mask_np * 255).astype(np.uint8)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return mask_np

        # Process each contour with Fourier Descriptors and reconstruct
        smoothed = np.zeros((H, W), dtype=np.uint8)

        for contour in contours:
            contour = contour.squeeze(1)  # (N, 2)
            if len(contour) < 4:
                continue

            # Represent contour as complex numbers: x + jy
            complex_contour = contour[:, 0].astype(np.float64) + 1j * contour[:, 1].astype(np.float64)

            # Apply FFT (Fourier Descriptors)
            fd = np.fft.fft(complex_contour)

            # Keep only num_descriptors low-frequency components
            # Zero out high-frequency components
            n = len(fd)
            keep = min(num_descriptors // 2, n // 2)

            fd_filtered = np.zeros_like(fd)
            fd_filtered[0:keep] = fd[0:keep]
            if keep > 0:
                fd_filtered[n - keep:] = fd[n - keep:]

            # Inverse FFT to reconstruct smoothed contour
            smoothed_contour = np.fft.ifft(fd_filtered)

            # Extract x, y coordinates
            xs = np.real(smoothed_contour)
            ys = np.imag(smoothed_contour)

            # Clip to image bounds
            xs = np.clip(np.round(xs).astype(np.int32), 0, W - 1)
            ys = np.clip(np.round(ys).astype(np.int32), 0, H - 1)

            # Reconstruct contour array for cv2.fillPoly
            reconstructed = np.stack([xs, ys], axis=1).reshape(-1, 1, 2).astype(np.int32)

            # Fill the smoothed contour
            cv2.fillPoly(smoothed, [reconstructed], 255)

        # Normalize back to [0, 1]
        return (smoothed / 255.0).astype(np.float32)


NODE_CLASS_MAPPINGS = {
    "MaskFourierSmoothing": MaskFourierSmoothing,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFourierSmoothing": "Mask Fourier Smoothing",
}

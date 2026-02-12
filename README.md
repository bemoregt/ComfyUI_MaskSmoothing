# ComfyUI Custom Node: Mask Fourier Smoothing

A ComfyUI custom node that smooths mask contours using the **Fourier Descriptor** algorithm. It takes a mask as input, applies low-pass filtering in the frequency domain, and outputs a smoothed mask.

## Example

| Input Mask | Smoothed Mask (descriptors=32) |
|---|---|
| Jagged, noisy contour | Clean, smooth contour |

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes/` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/bemoregt/ComfyUI_CustomNode_MaskSmoothing.git
```

2. Install dependencies:

```bash
pip install -r custom_nodes/ComfyUI_CustomNode_MaskSmoothing/requirements.txt
```

3. Restart ComfyUI.

The node will appear under **mask/processing** → **Mask Fourier Smoothing**.

## Usage

Connect a `MASK` output to the node's `mask` input. Adjust `num_descriptors` to control the smoothing strength.

```
[Load Image] ──► mask ──► [Mask Fourier Smoothing] ──► smoothed_mask ──► [Preview Mask]
```

## Parameters

| Parameter | Default | Range | Description |
|---|---|---|---|
| `mask` | — | — | Input mask (MASK type) |
| `num_descriptors` | 32 | 4 – 512 | Number of Fourier frequency components to keep. Lower = smoother contour. Higher = closer to original. |

## How It Works

The algorithm follows these steps:

1. **Binarize** the input mask (threshold at 0.5)
2. **Extract contours** using `cv2.findContours`
3. **Encode** contour points as complex numbers: `z = x + j·y`
4. **Apply FFT** to obtain the Fourier Descriptors
5. **Low-pass filter**: keep only the lowest `num_descriptors` frequency components (zero out the rest)
6. **Apply inverse FFT** to reconstruct the smoothed contour
7. **Fill** the reconstructed contour to produce the output mask

The key insight is that high-frequency Fourier components correspond to sharp corners and jagged edges, while low-frequency components capture the overall smooth shape. By discarding high-frequency components, the contour is effectively smoothed.

## Dependencies

- `numpy`
- `opencv-python`
- `torch`

## License

MIT License

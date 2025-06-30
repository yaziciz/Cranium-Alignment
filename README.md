# Cranium-Alignment

Cranium-Alignment is a Python toolkit for automatic alignment of head CT scans using deep learning-based landmark detection and geometric transformations. It aligns CT images based on anatomical landmarks (nasal bridge and cochleas), similar to the orbito-meatal line, making downstream analysis and visualization more robust and reproducible. The methods are specifically optimized for the postmortem cranium alignment.

---

## Features

- **Automatic landmark detection** using pre-trained MONAI models.
- **Robust alignment** via pitch, yaw, and roll correction.
- **Batch processing** for multiple CT scans.
- **Easy integration** with your own data and workflows.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yaziciz/Cranium-Alignment.git
    cd Cranium-Alignment
    ```

2. **Install dependencies:**
    ```bash
    conda env create -f requirements.yaml
    ```
    *(Make sure you have Python 3.8+ and [PyTorch](https://pytorch.org/) with CUDA if you want GPU acceleration.)*

---

## Usage

### 1. **Align a Single CT Scan**

You can use the `align` function in your own scripts:

```python
from functions import align

nii_path = "path/to/your/input_scan.nii.gz"
output_path = "path/to/output_aligned_scan.nii.gz"

align(nii_path, output_path, debug=True)
```

### 2. **Batch Processing (Recommended)**

Use the provided `main.py` script to align all `.nii.gz` files in the `tests/test_data` folder and save results to `tests/outputs`:

```bash
python main.py
```

### 3. **Folder Structure**

```
Cranium-Alignment/
│
├── aux.py
├── functions.py
├── main.py
├── models/
│   ├── ct-head-adult-lc-best-metric-model-*.pth
│   ├── ct-head-adult-nb-best-metric-model-*.pth
│   └── ct-head-adult-rc-best-metric-model-*.pth
├── tests/
│   ├── outputs/
│   └── test_data/
└── LICENSE
```

- Place your input `.nii.gz` files in `tests/test_data/`.
- Aligned outputs will appear in `tests/outputs/`.

---

## Model Files

The `models/` directory must contain the three pre-trained MONAI model files:
- `ct-head-adult-nb-best-metric-model-*.pth` (nasal bridge)
- `ct-head-adult-rc-best-metric-model-*.pth` (right canthus/cochlea)
- `ct-head-adult-lc-best-metric-model-*.pth` (left canthus/cochlea)

These are required for landmark detection.

The models are used from the [ct_head_align](https://github.com/radiplab/ct_head_align) repository.

---

## Example Script

```python
from functions import align

# Align a single scan
align("tests/test_data/sample.nii.gz", "tests/outputs/sample_aligned.nii.gz", debug=True)
```

---

## Troubleshooting

- **Landmark not found:** The script will attempt to refine spacing and retry. If it still fails, check your input image quality and orientation.
- **Pitch correction slightly off:** For difficult cases, the alignment is iteratively refined. If needed, run the alignment twice or increase the number of iterations in `functions.py`.
- **Missing models:** Ensure all three model files are present in the `models/` directory. To optimize the detection phase, single model could be trained for multi-landmark detection.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
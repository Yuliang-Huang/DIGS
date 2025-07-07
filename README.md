# DIGS

Code for the MICCAI 2025 paper:  
**[DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model](https://arxiv.org/abs/2506.22280)**

> ⚠️ **Note**: This repository is under active development and will be frequently updated over the coming months.

## Key Features

- **Dynamic Reconstruction**: Reconstructs CBCT volumes **at each projection**, eliminating the need for respiratory phase sorting.

- **Optimized Gaussian Splatting Backbone**: Builds on [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) with several engineering improvements, detailed [here](./digs/submodules/xray-gaussian-rasterization-voxelization/README.md).

- **Low-Rank FFD-Based Motion Representation**: Employs a low-rank Free-Form Deformation (FFD) model to represent motion in a compact and spatially consistent manner, achieving over **6× speedup** compared to HexPlane while delivering higher reconstruction quality.  
  *Note: This speedup reflects only the improvement from the motion representation and does **not** include additional acceleration from engineering enhancements to the Gaussian splatting backbone.*

- **Deformation-Informed Motion Modeling**: Introduces a unified deformation field that jointly drives the evolution of each Gaussian's **mean**, **scale**, and **rotation**, enforcing physical consistency throughout the reconstruction process.

## Installation

```bash
git clone --recursive https://github.com/Yuliang-Huang/DIGS.git
cd DIGS
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate digs
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

Then follow this [link](https://github.com/CERN/TIGRE/blob/master/Frontispiece/python_installation.md) to install TIGRE.

## Dataset

The dataset will be uploaded soon.

We use data in the [NAF](https://github.com/Ruyi-Zha/naf_cbct) format (`*.pickle`), with the following differences:

1. **Units** are in millimeters (mm) so no need to divide attenuation by 1000.  
2. **Axis order**:  
   - For volumes: `x`, `y`, `z`  
   - For detectors: `u` (horizontal), `v` (vertical)
3. **Angles** are stored directly as an array, rather than using `totalAngle` and `startAngle`.
4. **Attenuation values** are stored as real values, without normalization to the [0, 1] range.

## Training

### Step 1: Initialize Gaussians

Run the following command to initialize the Gaussians. The script is adapted from [this implementation](https://github.com/fuyabo/4DGS_for_4DCBCT/blob/7715e543f68936cce52228c22fd834b4dfafdaa4/initialize_pcd.py):

```bash
python initialize_pcd.py --data <path_to_pickle_file>
```

### Step 2: Fit the Gaussians
Use the following command to train the model with the deformation-informed framework:

```bash
python train.py -s <path_to_pickle_file> -m <path_to_output_folder> --unified --sag_slice <x> --cor_slice <y> --ax_slice <z>
```
*Note: 
- \<x\>, \<y\> and \<z\> specify the sagittal, coronal, and axial slices to be visualized in TensorBoard.
- If not specified, they default to 0.

To disable the deformation-informed framework:

```bash
python train.py -s <path_to_pickle_file> -m <path_to_output_folder> --sag_slice <x> --cor_slice <y> --ax_slice <z>
```

To further disable the free-form deformation (FFD) representation:

```bash
python train.py -s <path_to_pickle_file> -m <path_to_output_folder> --no_bspline --sag_slice <x> --cor_slice <y> --ax_slice <z>
```

## Evaluation

```bash
python test.py -m <path_to_output_folder>  --skip_render_train --skip_render_test --axis 1 0 --slices <> --unified
```
This command performs evaluation and reports PSNR and SSIM values.

It also saves videos of dynamic reconstructions at:
- Coronal slice (axis=1, slice index <y>)
- Sagittal slice (axis=0, slice index <x>)

The output videos will be saved to \<path_to_output_folder\>/test/.

*Note:
Be sure to include the correct options --unified and/or --no_bspline to match the configuration used during training.

## Citation

Please cite our paper if you find this repository useful for your research
```bib
@misc{huang2025digs,
      title={DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model}, 
      author={Yuliang Huang and Imraj Singh and Thomas Joyce and Kris Thielemans and Jamie R. McClelland},
      year={2025},
      eprint={2506.22280},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.22280}, 
}
``` 

## Acknowledgement

This repo is built on

https://github.com/ruyi-zha/r2_gaussian

https://github.com/fuyabo/4DGS_for_4DCBCT 

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

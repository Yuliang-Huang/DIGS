# DIGS

Code for the MICCAI 2025 paper:  
**[DIGS: Dynamic CBCT Reconstruction using Deformation-Informed 4D Gaussian Splatting and a Low-Rank Free-Form Deformation Model](https://arxiv.org/abs/2506.22280)**

---

## Key Features

- **Dynamic Reconstruction**: Reconstructs CBCT volumes **at each projection**, eliminating the need for respiratory phase sorting.

- **Optimized Gaussian Splatting Backbone**: Builds on [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) with several engineering improvements, detailed [here](./digs/submodules/xray-gaussian-rasterization-voxelization/README.md).

- **Low-Rank FFD-Based Motion Representation**: Employs a low-rank Free-Form Deformation (FFD) model to represent motion in a compact and spatially consistent manner, achieving over **6× speedup** compared to HexPlane while delivering higher reconstruction quality.  
  *Note: This speedup reflects only the improvement from the motion representation and does **not** include additional acceleration from engineering enhancements to the Gaussian splatting backbone.*

- **Deformation-Informed Motion Modeling**: Introduces a unified deformation field that jointly drives the evolution of each Gaussian's **mean**, **scale**, and **rotation**, enforcing physical consistency throughout the reconstruction process.

---

**Citation**

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

**Acknowledgement**

This repo is built on

https://github.com/ruyi-zha/r2_gaussian

https://github.com/fuyabo/4DGS_for_4DCBCT 

**License**

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

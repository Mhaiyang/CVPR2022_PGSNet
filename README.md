# CVPR2022_PGSNet

## Glass Segmentation using Intensity and Spectral Polarization Cues
[Haiyang Mei](https://mhaiyang.github.io/), [Bo Dong](https://dongshuhao.github.io/), Wen Dong, Jiaxi Yang, Seung-Hwan Baek, [Felix Heide](https://www.cs.princeton.edu/~fheide/), [Pieter Peers](http://www.cs.wm.edu/~ppeers/), Xiaopeng Wei, [Xin Yang](https://xinyangdut.github.io/)

[Paper] [[Project Page](https://mhaiyang.github.io/CVPR2022_PGSNet/index.html)]

- [Table of Contents](#glass-segmentation-using-intensity-and-spectral-polarization-cues)
  * [1. Abstract](#1-abstract)
  * [2. Requirements](#2-requirements)
  * [3. Experiments](#3-experiments)
    + [3.1. Train](#31-train)
    + [3.2. Test](#32-test)
  * [4. Results](#4-results)
    + [4.1. Qualitative Comparison](#41-qualitative-comparison)
    + [4.2. Quantitative Comparison](#42-quantitative-comparison)
    + [4.3. Results Download](#43-results-download)
  * [5. Proposed RGBP-Glass Dataset](#5-proposed-rgbp-glass-dataset)
    + [5.1. Overview](#51-overview)
    + [5.2. File Structure](#52-file-structure)
    + [5.3. Download](#53-download)
  * [6. Citation](#6-citation)
  * [7. LICENSE](#7-license)
  * [8. Contact](#8-contact)

### 1. Abstract


### 2. Requirements
* Python 3.8.10
* PyTorch == 1.10.0
* TorchVision == 0.11.0
* CUDA 11.4
* tqdm
* timm

Lower version should be fine but not fully tested :-)


### 3. Experiments

#### 3.1 Train
`python train.py`

#### 3.2 Test
Download 'Conformer_base_patch16.pth' at [here](https://drive.google.com/file/d/1UoOyGa-vQtGWLAl-VADJ1bedzMaAvc22/view?usp=sharing) and pre-trained model 'PGSNet.pth' at [here](https://mhaiyang.github.io/CVPR2022_PGSNet/index.html), then run `infer.py`.


### 4. Results

#### 4.1. Qualitative Comparison
<p align="center">
    <img src="assets/coming_soon.png"/> <br />
    <em> 
    Figure 1: Qualitative comparison results.
    </em>
</p>

#### 4.2. Quantitative Comparison (Overall/Sub-class)

<p align="center">
    <img src="assets/coming_soon.png"/> <br />
    <em> 
    Table 1: Quantitative comparison results.
    </em>
</p>

#### 4.3. Results Download 

1. Results of our SINet can be found in this [download link](https://drive.google.com/drive/folders/1uxIuT5havrkL07Skp_oQHoiJ8m2he1Fn?usp=sharing).

2. Performance of competing methods can be found in this [download link](https://drive.google.com/open?id=1jGE_6IzjGw1ExqxteJ0KZSkM4GaEVC4J).

### 5. Proposed COD10K Datasets

#### 5.1. Overview

<p align="center">
    <img width="600" height="200" src="assets/coming_soon.png"/> <br />
    <em> 
    Overview of our RGBP-Glass dataset.
    </em>
</p>

#### 5.2. File Structure
	RGBP-Glass
	├── EvaluationTool
	│   ├── CalMAE.m
	│   ├── Enhancedmeasure.m
	│   ├── Fmeasure_calu.m
	│   ├── main.m
	│   ├── original_WFb.m
	│   ├── S_object.m
	│   ├── S_region.m
	│   └── StructureMeasure.m
	├── Images
	│   ├── CamouflagedTask.png
	│   ├── CamouflagingFromMultiView.png
	│   ├── CmpResults.png
	│   ├── COD10K-2.png
	│   ├── COD10K-3.png
	│   ├── COVID'19-Infection.png
	│   ├── locust detection.png
	│   ├── new_score_1.png
	│   ├── PolypSegmentation.png
	│   ├── QuantitativeResults-new.png
	│   ├── SampleAquaticAnimals.png
	│   ├── Search-and-Rescue.png
	│   ├── SINet.png
	│   ├── SubClassResults-1.png
	│   ├── SubClassResults.png
	│   ├── Surface defect Detection2.png
	│   ├── TaskRelationship.png
	│   ├── Telescope.png
	│   └── UnderwaterEnhancment.png
	├── MyTest.py
	├── README.md
	├── requirement.txt
	└── Src
	    ├── backbone
	    ├── __init__.py
	    ├── SearchAttention.py
	    ├── SINet.py
	    └── utils

#### 5.3. Download

<p align="center">
    <img src="assets/coming_soon.png"/> <br />
    <em> 
    </em>
</p>

### 6. Citation
Please cite our paper if you find the work useful::

```
@InProceedings{Mei_2022_CVPR_PGSNet,
    title = {Glass Segmentation using Intensity and Spectral Polarization Cues},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2022}
}
```


### 7. License

[//]: # (Please see `LICENSE`)

- The RGBP-Glass Dataset is made available for non-commercial purposes only.

- You will not, directly or indirectly, reproduce, use, or convey the COD10K Dataset 
or any Content, or any work product or data derived therefrom, for commercial purposes.

This code is for academic communication only and not for commercial purposes. 
If you want to use for commercial please contact me.

Redistribution and use in source with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in
  the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 	
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

### 8. Contact
E-Mail: mhy666@mail.dlut.edu.cn


**[⬆ back to top](#1-abstract)**
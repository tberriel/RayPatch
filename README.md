# Ray-Patch: An Efficient Decoder for Light Field Transformers

Official implementation of the paper ["Ray-Patch: An Efficient Decoder for Light Field Transformers"](https://arxiv.org/abs/2305.09566).

<img src="https://drive.google.com/uc?export=view&id=11Ol27ifHZihLYiM157XpwCZD_zYIGOMM" alt="Querying comparison" width="512"/>
<img src="https://drive.google.com/uc?export=view&id=1-0clAkYwOCGMF0BOM71ij9Tv0NUJg6Q7" alt="Architecture" width="512"/>

## Results
### MSN-Easy
|Run | PSNR | SSIM | LPIPS | Rendering Speed | Download |
|---|---|---|---|---|---|
|`srt` |31.19 | 0.904 | 0.156 | 114 fps |[Link](https://drive.google.com/file/d/18xN2D5PNHpWIBoV28ylzWln3cMW71l5u)|
|`RP-srt k=2` |31.16| 0.904 | 0.147 | 231 fps |[Link](https://drive.google.com/file/d/1q1KQ47EzhJflB9QTgAIKJb6R3vnLb-oK)|
|`RP-srt k=4` |30.89| 0.898 | 0.166 | 280 fps |[Link](https://drive.google.com/file/d/1ssTLIRKFRL4923uG6lxuUFnvMDkfvlVL)|
|`RP-srt k=8` |30.16| 0.879 | 0.209 | 275 fps |[Link](https://drive.google.com/file/d/16CSyko7DI3rM2WLQIdQGBC_GSN2S579T)|


### ScanNet
|Run | PSNR | SSIM | LPIPS | RMSE | Abs.Rel. | Square Rel.| Rendering Speed | Download |
|---|---|---|---|---|---|---|---|---|
|`DeFiNe`| 23.46 | 0.783 | 0.495 | 0.275 | 0.108 | 0.053 | 7 fps |[Link](https://drive.google.com/file/d/16CSyko7DI3rM2WLQIdQGBC_GSN2S579T)|
|`RP-DeFiNe k=16`| 24.54 | 0.801 | 0.453 | 0.263 | 0.103 | 0.050 | 208 fps |[Link](https://drive.google.com/file/d/1dHyWDXGYsRx9cpe93hslr8gDKwoi37yq)|


## Setup
The implementation has been done using Pytorch 1.11.0, Pytorch Lightning 1.7.3, and cuda 11.3.
To run the repository we suggest to use the conda environment:

 * Clone the repository
    ``` 
    git clone git@github.com:tberriel/RPDecoder_private.git
    ```
 * Create a conda environment
    ``` 
    conda env create -n RayPatch --file=raypatch.yml 
    conda activate RayPatch 
    ```

### Data
The models are evaluated on two datasets:
* MultiShapeNet-Easy dataset, introduced by [Stelzner et al.](https://stelzner.github.io/obsurf/): Download from [Link](https://drive.google.com/file/d/1RlHIbJ9NDtFgDBs1v0oQNhirXJak04UV)
* ScanNet dataset [Dai et al.](https://github.com/ScanNet/ScanNet): Follow the original repository instructions to acces the dataset. Then, to decode [NASDE](https://github.com/udaykusupati/Normal-Assisted-Stereo) stereo pairs used for training and evaluation by DeFiNe, follow these intructions:
    * After downloading ScanNet data, uncompress it with our modified scripts:
      ```
        cp /<path to rpdecoder>/src/SensReader/* /<path to scannet>/ScanNet/SensReader/.
        cd /<path to scannet>/ScanNet/SensReader
        python decode.py --dataset_path /<path to scannet>/scans --output_path /<path to scannet>/data/val/ --split_file scannetv2_val.txt --frames_list frames_val.txt
        python decode.py --dataset_path /<path to scannet>/scans --output_path /<path to scannet>/data/train/ --split_file scannetv2_train.txt --frames_list frames_train.txt
      ```
    * Then run the following script to preprocess it:
      ```
        cd /<path to rpdecoder>/
        python srt/data/preproces_scannet.py /<path to scannet>/data/ /<path to rpdecoder>/data/scannet/ --parallel --num-cores 12
        mv /<path to rpdecoder>/data/stereo_pairs_* /<path to rpdecoder>/data/scannet/.
      ```
      Preprocessing consist of resizing input RGB data to 480x640 resolution. Set ``` --num-cores ``` to the number of cores of your cpu to process multiple scenes in parallel.
      
Ensure the data is placed in their respective folders:
  ```
  |-- rpdecoder
     |-- data
       |-- msn_easy
          |-- train
          |-- val
          |-- test
       |-- scannet
          |-- train
          |-- val
  ```
## Experiments
Each training run should be stored inside the runs folder of the respective dataset, with its corresponding configuration file:
  ```
  |-- rpdecoder
    |-- runs
        |-- scannet
          |-- define_32_stereo_acc
              |-- config.yaml
              |-- model_best.ckpt
          |-- rpdefine_16_32_stereo_acc
              |-- config.yaml
              |-- model_best.ckpt
  ```
### Test
To evaluate a model run:
```
  python test.py /<path to config file>/
```
Add flag `--vis` to render a batch of images. Use flag `--num_batches` to set the number of batches to save.

By defalut evaluation does not compute neither LPIPS nor SSIM. To compute them add respective flags: 
```
  python test.py /<path to config file>/ --lpips --ssim
```
SSIM computation has a huge memory footprint. To evaluate `define_stereo_32` we had to run evaluation on CPU with 160 GB of RAM.

To evaluate profiling metrics for render a single image run:
```
python test_profile.py <path to config folder> --batch --flops --time
```
The `<path to config folder>` should be like `runs/scannet/define_32_stereo_acc/`.

To execute on a GPU device add `--cuda` flag.

### Train
To train a model run:
```
  python train.py /<path to config file>/ 
```

Training the model also has a huge memory footprint. We trained both models using 4 Nvidia Tesla V100 with 32 GBytes of VRAM each. We used batch 16 and gradient accumulation of 2 to simulate batch size 32.

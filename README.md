# Pytorch Contrast Adaptive Sharpening

This repository is an unofficial PyTorch implementation of the [Contrast Adaptative Sharpening](https://github.com/GPUOpen-Effects/FidelityFX-CAS/tree/master) (CAS) featured in AMD's FidelityFX.
It is designed to be the lightweight final step of bigger image processing pipelines, as it attenuates the blur typically introduced by image upscaling while minimizing artifacts.
Below, you will find information about the code, installation requirements, and how to run the provided example.

## Illustration

Here is a quick overview of the sharpening on a famous image of the [Kodak dataset](https://www.r0k.us/graphics/kodak/):

![Feature Illustration](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/blob/main/data/illustration.gif)

_Description: A blurry image is sharpen with 2 different strengh. Notice how the clouds and distant mountain stay untouched_

## Requirements

PyTorch is the only requirement. You can install it using the following command:

```
pip install torch
```


Make sure to install a compatible version of PyTorch based on your system and preferences.

## Example

To run the provided example, follow these steps:

1. Make sure you have fulfilled the requirements mentioned above by installing PyTorch.

2. Additionally, the example script `example.py` utilizes Matplotlib for visualizations. To install Matplotlib, you can use the following command:

```
pip install matplotlib
```

3. Once you have PyTorch and Matplotlib installed, you can run the example script using the following command:

```
python example.py
```

## Issues and Contributions

If you encounter any issues while using this code or have suggestions for improvements, please feel free to open an issue on this repository. Contributions are also welcome! Please follow the standard GitHub workflow for making pull requests.

## Contact

If you have any questions or encounter an issue, you can contact me at [jamy.lafenetre@ens-paris-saclay.fr].


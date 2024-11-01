# Bolt_24D0894_IE643
This repository has been created to submit the project work in meeting the requirements of coursework of IE643 at IITB
<br>
Team Name is 24D0894_BOLT
<br>
Author - Apurva Mayank (24D0894)
# Unsupervised Anomaly Segmentation in Brain MRI/CT Scans

## Problem Statement
The objective of this project is to develop an unsupervised deep learning model for anomaly segmentation in brain MRI/CT scans, specifically targeting tumor or other abnormal regions using single-sample images. This work addresses the challenge of detecting anomalies in a highly constrained setup where only a single brain scan is available for model training, and no annotations or additional samples are provided. Given these constraints, the model must adapt and learn to segment anomalous regions independently on each image without relying on pre-trained weights or labeled data.

The project requirements include:
1. Collecting a brain tumor dataset, performing necessary preprocessing, and extracting 2D scan images.
2. Developing deep learning approaches for anomaly detection, where the model performs segmentation on single-sample images without annotated anomalies. The model should learn independently with each new scan and detect anomalous regions accordingly.
3. Experimenting with the proposed models on the collected dataset.
4. Creating an interface for real-time detection of anomalous regions in test images.

## Approaches Taken

To achieve the project objectives, two primary deep learning approaches were explored: **AnoGAN (Anomaly GAN)** and **Convolutional Autoencoder (ConvAutoencoder)**. Both approaches were designed to detect anomalies in a single brain scan by identifying regions that deviate from "normal" brain tissue patterns. Details of the data preprocessing, model architectures, and training procedures are documented in the following core notebooks.

### 1. Data Preprocessing
The data preprocessing workflow is available in the notebook **Data_preprocessing_first_experiment_with_GAN.ipynb**. Key steps include:
- **Dataset**: We used the BRATS 2020 dataset, a multi-modal brain MRI dataset in NIfTI format.
- **Slice Selection**: For each 3D scan, the slice with the highest number of non-zero pixels was selected. This slice likely contains the most diagnostic information.
- **Normalization and Conversion**: The selected slice was resized to 256x256 pixels, normalized to the range [0, 1], and saved as an `.npy` file. This file format is compatible with the models used in the project and allows for consistent data handling.

### 2. AnoGAN for Anomaly Detection
The AnoGAN approach is implemented in **AnoGAN_24D0894.ipynb**. 

#### Model Description
- **Architecture**: AnoGAN uses a **generator** and **discriminator** network. The generator attempts to recreate "normal" brain structures, while the discriminator evaluates the quality of these reconstructions, guiding the generator to produce realistic images.
- **Latent Space Optimization**: For each input, latent space optimization is performed to adjust the latent vector for the most accurate reconstruction. This difference between input and reconstructed images highlights anomalous regions.
  
#### Training Procedure
- **Optimizer**: Adam optimizer with a learning rate of 0.0001 for both generator and discriminator.
- **Epochs**: Up to 500 epochs with early stopping if no further improvement was observed.
  
#### Challenges and Adjustments
- **Overfitting**: Mode collapse and overfitting were mitigated by simplifying the generator architecture.
- **High Computational Cost**: The latent space optimization for each image required high computational resources, which was partially resolved by reducing the latent space dimensions.

### 3. Convolutional Autoencoder for Anomaly Detection
The Convolutional Autoencoder approach is implemented in **mri_autoencoder.ipynb**.

#### Model Description
- **Architecture**: The ConvAutoencoder comprises an **encoder** and **decoder**. The encoder compresses the input into a latent space, while the decoder reconstructs the image. Anomalies are identified through reconstruction errors.
  
#### Training Procedure
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Epochs**: Up to 300 epochs, with early stopping based on loss stabilization.

#### Anomaly Detection
The reconstruction error highlights regions that deviate from the model’s learned "normal" pattern, with dynamic thresholding applied to segment anomalous areas.

### 4. Data Augmentation
To improve model generalization, data augmentation (random rotations and flips) was applied to the `.npy` files. This simulated variability in data and reduced the risk of overfitting in both AnoGAN and ConvAutoencoder models.

## Hardware Configuration
The experiments were conducted on a system with the following specifications:
- **CPU**: Intel Xeon E5-2678
- **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
- **RAM**: 64 GB
- **OS**: Ubuntu 18.04 LTS
- **Framework**: PyTorch

## Repository Structure
- **Data_preprocessing_first_experiment_with_GAN.ipynb**: Preprocessing steps for BRATS 2020 data, including slice selection and conversion to .npy format.
- **AnoGAN_24D0894.ipynb**: Implementation of AnoGAN model for anomaly detection with training and evaluation steps.
- **mri_autoencoder.ipynb**: Implementation of the Convolutional Autoencoder model, including anomaly detection via reconstruction error.

## GitHub Repository Link
The entire project code, including data preprocessing scripts, model implementations, and training routines, can be accessed here:
[https://github.com/ApurvaMayank-iitb/24D0894_IE643_NIFTI](https://github.com/ApurvaMayank-iitb/24D0894_IE643_NIFTI)

## References
- BRATS Dataset: [https://www.med.upenn.edu/cbica/brats2020/](https://www.med.upenn.edu/cbica/brats2020/)
- GAN Models: [https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
- NIfTI Preprocessing: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

For further details on the approach, experiment settings, and model evaluation, please refer to the notebooks in this repository.

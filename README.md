# UNet-for-Helmotz-Equation

From Deep Learning to Brain Stimulation: Optimizing Transcranial Ultrasound Focusing with Time Reversal Techniques

## Repository Structure

This repository contains several variations of the U-Net model for solving the Helmholtz equation, with modifications aimed at exploring different loss functions and activation functions.

### File and Folder Descriptions

1. **run.py**:  
   - This script contains the initial U-Net structure designed for this project. It uses the standard setup and configuration, with results saved in the `L2` folder.

2. **L2 Folder**:  
   - Contains the results generated by `run.py` with the original U-Net model using the L2 loss function.

3. **H1 Folder**:  
   - Contains a modified version of the U-Net code that implements the H1 loss instead of the L2 loss. This adjustment is intended to explore how the change in loss function impacts the model's performance in solving the Helmholtz equation.

4. **L2_v2 and L2_v3 Folders**:  
   - These folders contain versions of the U-Net where the activation functions have been modified from the original structure. `L2_v2` and `L2_v3` represent different variations in the activation functions, providing insights into the effect of these changes on the model's output.

## How to Use

- **Running the Model**:  
   To run the initial model, use `run.py`. This will generate outputs and save them in the respective folders based on the configuration.
  
- **Switching Loss Functions**:  
   To explore the H1 loss implementation, refer to the code in the `H1` folder.
  
- **Exploring Activation Function Variations**:  
   For changes in activation functions, check the `L2_v2` and `L2_v3` folders, which contain alternative configurations.

---

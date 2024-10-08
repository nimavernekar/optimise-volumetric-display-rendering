
# Opimising Acoustic Volumteric Displays for Accurate rendering of leviatted particles: A data driven approach

This repository contains the code and models used to **optimize the rendering of levitated particles in acoustophoretic volumetric displays** using a data-driven approach. The focus is on reducing the deviation between the simulated and actual measured positions of levitated particles. Two neural network models are trained: one to predict the particle’s measured positions and theta, and another to optimize trap positions using a custom cost function. The project is structured into different notebooks representing various versions of the models.

## Project Structure

- **Data Cleaning**: Initial data cleaning is performed to ensure consistency and accuracy in the dataset. The steps involve handling missing data, reformatting the data to match the required input formats for neural networks, and preparing the data for training and testing.
  
- **Model Versions**: There are three main versions of the models that differ in how the cost function is calculated within the Model 2. These versions are contained within six Python notebooks.

### Notebooks

1. **Model Architecture Summary**:
    - **Model 1**: A LSTM-based network with 64 units and dense layers. It predicts measured particle positions (`measured_y`, `measured_z`) and theta based on input trap positions.
    - **Model 2**: An LSTM-based network with 64 units and dense layers. This model utilizes a custom loss function designed to minimize the deviation between predicted measured positions gathered from Model 1 and actual reference positions to finally predicting optimized trap positions.
  
2. **Key Metrics**:
    - **Model 1**:
        - **Training Time**: X seconds
        - **Inference Time**: Y seconds per sample
        - **R-Squared**: Z
        - **MSE**: A
        - **MAE**: B
    
    - **Model 2**:
        - **Training Time**: X seconds
        - **Inference Time**: Y seconds per sample
        - **R-Squared**: Z
        - **MSE**: A
        - **MAE**: B

3. **Loss Evolution**:
    - Plots are provided to showcase the **evolution of training and validation loss** during model training.
    - **Model 1**: Training vs. Validation Loss
    - **Model 2**: Training vs. Validation Loss (incorporating the custom loss function)

4. **Prediction Comparisons**:
    - **Model 1 Predictions**:
        - Scatter plots comparing the predicted and actual values for `measured_y`, `measured_z`, and theta.
    - **Model 2 Predictions**:
        - Comparison of the predicted reference positions (`traps_y`, `traps_z`) against actual reference positions.

5. **Visualization of Results**:
    - Includes scatter plots and visualizations for comparing the performance of the two models.
    - Error density plots and **predicted vs actual position comparisons** are used to highlight model accuracy.

## Requirements

- Python 3.x
- TensorFlow/Keras
- Pandas
- NumPy
- Matplotlib

## Key objectives include:
- Reducing error between real and simulated particle paths
- Improving the accuracy of particle positioning in 3D space
- Leveraging deep learning models to enhance volumetric display rendering

## Technologies and tools used:
- Python (TensorFlow, Keras)
- Jupyter Notebooks for experimentation
- Data preprocessing and transformation for time-series analysis
- Git for version control

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Cleaning**: Start with the data cleaning process in the provided notebook to ensure the dataset is ready for training.
   
2. **Train the Models**: Train the GRU-based and LSTM-based networks for particle prediction and optimization.
   - **Model 1** predicts the measured particle positions.
   - **Model 2** optimizes trap positions using a custom loss function.

3. **Run Predictions**: Compare the predictions for measured and reference positions with the actual positions. Use scatter plots and metrics like MSE and R-squared to evaluate model performance.

4. **Visualizations**: Generate visualizations for loss evolution, prediction comparisons, and error analysis.

## Results

- The LSTM-based network (Model 1) predicts measured positions, while the LSTM-based network (Model 2) optimizes trap positions with a custom loss function that incorporates predictions from Model 1.
- The results include improved accuracy in particle positioning, with reduced deviation between predicted and actual reference positions.
- Visualization plots demonstrate the effectiveness of the models and how the custom cost function influences the optimization process.

## Acknowledgments

- Special thanks to my supervisors and colleagues for their guidance and support throughout this project.

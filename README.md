# SVM Classification on Iris Dataset

## Overview

This project demonstrates how to build a Support Vector Machine (SVM) classifier using the Iris dataset, a classic dataset in pattern recognition and machine learning. The workflow follows a structured approach, from data exploration to model evaluation, leveraging Python's scientific libraries and scikit-learn.

<p align="center">
  <img src="1" alt="SVM Classification on Iris Dataset Workflow" width="350"/>
</p>

## Workflow

1. **Start**
2. **Import Libraries**  
   Import essential Python libraries including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
3. **Load Iris Data**  
   Load the Iris dataset using scikit-learn's built-in utilities.
4. **Create DataFrame and Explore Data**  
   Convert the dataset to a Pandas DataFrame for easy exploration and visualization.
5. **Preprocess Data**  
   Prepare the data for modeling, including feature selection and train-test splitting.
6. **Train SVM Model with Polynomial Kernel**  
   Build and train an SVM classifier using a polynomial kernel to capture non-linear relationships.
7. **Predict**  
   Use the trained model to make predictions on the test set.
8. **Evaluate Model**  
   Assess model performance using accuracy and visualization of decision boundaries.
9. **End**

## File Structure

- **SVM_Iris_data_project_1.ipynb**  
  Jupyter notebook containing step-by-step code implementation of the workflow above.
- **README.md**  
  This documentation file.
- **Workflow Image**  
  Visual reference for the project workflow.

## Key Features

- **Exploratory Data Analysis:**  
  Visualize the distribution of Iris features and classes.
- **Multiple Kernels:**  
  Experiment with polynomial, linear, and RBF kernels.
- **Decision Boundary Visualization:**  
  Use `mlxtend.plotting` to visually interpret the classifier’s decision regions.
- **Model Evaluation:**  
  Compare accuracy across kernels and visualize performance.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `mlxtend`

### Installation

Install the necessary libraries via pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn mlxtend
```

### Running the Notebook

1. Open `SVM_Iris_data_project_1.ipynb` in Jupyter Notebook or Google Colab.
2. Step through each cell to execute the workflow and view outputs.

## Results

- Achieved high accuracy (>75%) with polynomial kernel SVM on the Iris dataset.
- Visualized the effect of different kernels on the classification decision boundaries.
- Provided clear EDA and workflow for reproducibility.

## Author

**ABU HUZAIFA ANSARI**

## License

This project is licensed for educational and research use.

---

For any questions or improvements, feel free to open an issue or submit a pull request.

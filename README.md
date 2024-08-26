# Universal Prediction Model

## Overview
This project involves the development of a universal prediction model capable of predicting missing attributes in datasets across various categories. The model utilizes a Gradient Boosting Classifier and robust preprocessing techniques to ensure high accuracy and reliability.

## Features
- **Universal Applicability**: Designed to handle datasets with different types of data, including both numeric and categorical features.
- **Robust Preprocessing**: Includes imputation for missing values and scaling for numeric features, as well as encoding for categorical features.
- **High Accuracy**: The model achieves a minimum of 80% accuracy on the validation set, ensuring trustworthy predictions.
- **Flexible Input Handling**: Can adapt to datasets with missing features by assigning default values during prediction.

## Getting Started

### Prerequisites
- Python 3.7+
- pandas
- scikit-learn

### Installation
Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```

### Usage
1. **Prepare your datasets**: Ensure that your training and test datasets are in CSV format.
2. **Run the Model**:
    ```bash
    python gst.py
    ```
3. **Output**: Predictions for missing values in the test dataset will be saved to `test_predictions.csv`.

### Example
```python
# Load data
train_df = pd.read_csv('Booktrain.csv')
test_df = pd.read_csv('Booktest.csv')

# Build and train the model
target_column = 'Target (GST Category)'
model, numeric_features, categorical_features = build_and_train_model(train_df, target_column)

# Predict missing values on the test set
predict_missing_values(model, test_df, numeric_features, categorical_features)
```

## Model Structure

- **Preprocessing**:
  - **Numeric Features**: Imputation using the median value and standard scaling.
  - **Categorical Features**: Imputation using the most frequent value and one-hot encoding.
  
- **Model**: Gradient Boosting Classifier

## Performance
- **Validation Accuracy**: 1.00 (Example result; actual accuracy may vary based on the dataset).

## Contributing
Feel free to fork this repository and contribute by submitting a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)

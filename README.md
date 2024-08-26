# Universal Prediction Model

Welcome to the **Universal Prediction Model** repository! This project focuses on predicting missing attributes in a dataset across multiple categories with a minimum accuracy of 80%. The model is designed to be reliable, feasible, and trustworthy, utilizing advanced algorithms and techniques to ensure high-quality predictions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is designed to handle various types of datasets, making predictions on missing attributes with high accuracy. The model uses a combination of preprocessing techniques, feature engineering, and machine learning algorithms to deliver reliable predictions that can be applied to a wide range of datasets.

## Features

- **Universal Application**: The model is versatile and can be applied to any category of datasets.
- **High Accuracy**: Achieves a minimum accuracy of 80% on validation datasets.
- **Advanced Algorithms**: Utilizes Gradient Boosting, a powerful ensemble learning technique.
- **Scalable**: Designed to work efficiently with large datasets.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/Kushalsharma0702/prediction.ai.git
```

Navigate to the project directory:

```bash
cd prediction.ai
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your dataset**: Ensure your dataset is in CSV format and includes a target column that you want to predict.
2. **Update the configuration**: Modify the script to point to your dataset file.
3. **Run the model**: Execute the script to train the model and make predictions on your dataset.

```bash
python gst.py
```

## Model Overview

The Universal Prediction Model is built using the following components:

- **Data Preprocessing**: Handles missing values and scales numerical data.
- **Feature Engineering**: Encodes categorical data using OneHotEncoder.
- **Model Training**: Trains a Gradient Boosting Classifier for categorical predictions.
- **Prediction**: Generates predictions for missing attributes and outputs them to a CSV file.

## Contributing

We welcome contributions to enhance the model's performance, add new features, or improve documentation. Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

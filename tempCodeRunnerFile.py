import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

def build_and_train_model(train_df, target_column):
    # Define features and target
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Prepare target and features
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Encode the target variable if it is categorical
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a pipeline with preprocessor and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', GradientBoostingClassifier())])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Validate the model
    y_pred = pipeline.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Accuracy on validation set: {accuracy:.2f}")

    return pipeline, numeric_features, categorical_features

def predict_missing_values(model, test_df, numeric_features, categorical_features):
    # Ensure the test data has the same columns as the training data
    for feature in numeric_features:
        if feature not in test_df.columns:
            test_df[feature] = 0  # Default value for numeric columns
    
    for feature in categorical_features:
        if feature not in test_df.columns:
            test_df[feature] = 'unknown'  # Default value for categorical columns

    # Ensure columns are in the correct order
    test_df = test_df[numeric_features + categorical_features]

    # Predict on the test set
    test_predictions = model.predict(test_df)

    # Save predictions to CSV
    test_df.loc[:, 'Predictions'] = test_predictions
    test_df.to_csv('test_predictions.csv', index=False)

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('Booktrain.csv')
    test_df = pd.read_csv('Booktest.csv')

    # Build and train the model
    target_column = 'Target (GST Category)'
    model, numeric_features, categorical_features = build_and_train_model(train_df, target_column)

    # Predict missing values on the test set
    predict_missing_values(model, test_df, numeric_features, categorical_features)

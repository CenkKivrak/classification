import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def classify_dataset(filepath, target_col):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Basic preprocessing: drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # For simplicity, convert categorical variables to numeric via one-hot encoding
    X = pd.get_dummies(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, 
                         columns=[f'Predicted {cls}' for cls in sorted(y.unique())],
                         index=[f'Actual {cls}' for cls in sorted(y.unique())])
    
    # Calculate correct and missed counts
    correct = (y_pred == y_test).sum()
    missed = (y_pred != y_test).sum()
    
    return accuracy, cm_df, correct, missed

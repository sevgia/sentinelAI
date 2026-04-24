import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_adult_data():
    """
    Fetches the UCI Adult Census dataset and prepares it for Sentinel-AI.
    Target: Salary >50K (Privacy Sensitive)
    Protected Attributes: Sex, Race (Fairness Sensitive)
    """
    print("--- Fetching UCI Adult Census Data ---")
    data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    df = data.frame.copy()

    # 1. Force categorical columns to object/string first to break the 'Category' lock
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(object) # This fixes the 'dtype incompatible' error

    # 2. Now encode them
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df.loc[:, col] = le.fit_transform(df[col].astype(str))
        
    # 3. Final check: ensure the whole dataframe is numeric
    df = df.apply(pd.to_numeric)

    # 4. Identify key columns for Trustworthy AI
    target = 'class' # Income >50K
    protected_attrs = ['sex', 'race']

    protected_attrs_idx = [df.columns.get_loc(attr) for attr in protected_attrs]
    # 5. Basic Cleaning
    df = df.dropna()
    
    # 6. Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['category', 'object']).columns:
        df.loc[:, col] = le.fit_transform(df[col])
        
    # 8. Split Features and Target
    X = df.drop(columns=[target])
    y = df[target]
    
    # 9. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    # 9b. Capture RAW labels before scaling for the Fairness Audit
    # We only need the test set labels for the audit
    raw_gender_test = X_test.iloc[:, protected_attrs_idx[0]].values
    raw_race_test = X_test.iloc[:, protected_attrs_idx[1]].values

    # 10. Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Return the scaled data AND the raw audit labels
    return X_train_scaled, X_test_scaled, y_train, y_test, raw_gender_test, raw_race_test
    

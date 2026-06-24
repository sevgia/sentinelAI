import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_adult_data():
    """
    Fetches the UCI Adult Census dataset and prepares it for Sentinel-AI.
    Target: Salary >50K (Privacy Sensitive)
    Protected Attributes: Sex, Race (Fairness Sensitive)
    Now returns training-phase protected attributes for Adversarial Debiasing.
    """ 
    print("--- Fetching UCI Adult Census Data ---")

    data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    df = data.frame.copy()

    # 1. Break the 'Category' lock
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(object)

    # === FIX: MAP TO BINARY WHILE THEY ARE STILL RAW STRINGS ===
    df['race'] = df['race'].astype(str).str.strip().apply(lambda x: 1 if x == 'White' else 0)
    df['sex'] = df['sex'].astype(str).str.strip().apply(lambda x: 1 if x == 'Male' else 0)
    # ============================================================

    # 2. Encode the REMAINING text features
    le = LabelEncoder()
    # This loop will now skip or safely ignore processing 'sex' and 'race' if they are already numbers,
    # but to be safe, let's only encode columns that are still objects:
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['sex', 'race']:
            df.loc[:, col] = le.fit_transform(df[col].astype(str))
        
    df = df.apply(pd.to_numeric).dropna()

    # Identify key columns
    target = 'class'
    protected_gender = 'sex'
    protected_race = 'race'


    # 3. Split Features and Target
    X = df.drop(columns=[target])
    y = df[target]
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. CAPTURE PROTECTED ATTRIBUTES FOR BOTH SETS
    # These will be used by the Adversary during training to 'unlearn' bias
    gender_train = X_train[protected_gender].values
    race_train = X_train[protected_race].values
    
    gender_test = X_test[protected_gender].values
    race_test = X_test[protected_race].values

    # 6. Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Updated return signature to support Mitigation (training) and Audit (testing)
    return (X_train_scaled, X_test_scaled, 
            y_train, y_test, 
            gender_train, gender_test, 
            race_train, race_test)
    

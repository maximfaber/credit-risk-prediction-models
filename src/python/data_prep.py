import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv(r"/home/comp/Downloads/german_credit.csv")

    target = 'Creditability'
    numerical_features = ['Age (years)', 'Duration of Credit (month)', 'Credit Amount']

# (to be encoded)
    nominal_cats = [
        'Purpose', 'Foreign Worker', 'Sex & Marital Status', 'Guarantors',
        'Most valuable available asset', 'Type of apartment'
    ]

    ordinal_cats = [
        'Account Balance', 'Payment Status of Previous Credit', 'Value Savings/Stocks',
        'Length of current employment', 'Concurrent Credits'
    ]

    # Apply one-hot encoding to nominal categorical variables
    df_encoded = pd.get_dummies(df, columns=nominal_cats, drop_first=True)

    # Convert boolean columns to integer
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    # Get one-hot encoded column names
    one_hot_columns = [
        col for col in df_encoded.columns
        if any(nom in col for nom in nominal_cats)
    ]

    final_features = numerical_features + ordinal_cats + one_hot_columns

    # Select features and target
    X = df_encoded[final_features]
    y = df_encoded[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_features] = X_train[numerical_features].astype(float)
    X_test[numerical_features] = X_test[numerical_features].astype(float)

    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
    return X_train, X_test, y_train, y_test, X, y


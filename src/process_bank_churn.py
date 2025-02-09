import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple, Dict, Optional


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops unnecessary columns from the dataset.

    Args:
        df (pd.DataFrame): The raw input data.

    Returns:
        pd.DataFrame: DataFrame after dropping unnecessary columns.
    """
    return df.drop(columns=columns, errors='ignore')


def split_features_target(df: pd.DataFrame, input_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataset into features (X) and target (y).

    Args:
        df (pd.DataFrame): The raw input data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target data.
    """
    input_cols = df[input_cols].columns.tolist()
    return df[input_cols].copy(), df[target_col].copy()


def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target dataset.

    Returns:
        Tuple containing train and test splits for features and target.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler: Optional[MinMaxScaler]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Scales numeric features using MinMaxScaler.

    Args:
        X_train (pd.DataFrame): Training feature dataset.
        X_test (pd.DataFrame): Testing feature dataset.
        scaler (Optional[MinMaxScaler]): Scaler instance or None.

    Returns:
        Tuple containing scaled training and testing datasets along with the scaler.
    """
    if not scaler:
        return X_train, X_test, None

    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler


def encode_categorical_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Encodes categorical features using OneHotEncoder.

    Args:
        X_train (pd.DataFrame): Training feature dataset.
        X_test (pd.DataFrame): Testing feature dataset.

    Returns:
        Tuple containing transformed training and testing datasets along with the encoder.
    """
    categorical_cols = X_train.select_dtypes(
        include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    X_train = pd.concat([X_train.drop(columns=categorical_cols), pd.DataFrame(
        X_train_cat, index=X_train.index, columns=encoded_cols)], axis=1)
    X_test = pd.concat([X_test.drop(columns=categorical_cols), pd.DataFrame(
        X_test_cat, index=X_test.index, columns=encoded_cols)], axis=1)

    return X_train, X_test, encoder


def with_surname_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new features based on the raw data frame, specifically for the 'Surname' column and related attributes.

    This function adds the following features to the DataFrame:
    1. 'AgeAtFirstInteraction': Calculated as the difference between 'Age' and 'Tenure'.
    2. 'SurnameLength': The length of the surname (number of characters).
    3. 'SurnameSymbol': A combination of the first two characters of the surname.

    Args:
        raw_df (pd.DataFrame): The input DataFrame containing the customer data, including 'Age', 'Tenure', and 'Surname'.

    Returns:
        pd.DataFrame: The modified DataFrame with the new features added.
    """
    raw_df['AgeAtFirstInteraction'] = raw_df['Age'] - raw_df['Tenure']
    raw_df['SurnameLength'] = raw_df['Surname'].apply(len)
    raw_df['Surname'] = raw_df['Surname'].apply(lambda x: x[0] + x[1] if len(x) > 1 else x[0])

    return raw_df


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool = False) -> Dict[str, object]:
    """
    Preprocesses the raw dataset for training and testing.

    Args:
        raw_df (pd.DataFrame): The raw input data.
        scaler_numeric (bool): Whether to scale numerical features.

    Returns:
        Dict[str, object]: Processed data including train/test splits, scaler, and encoder.
    """
    df = with_surname_features(drop_columns(
        raw_df.copy(), ["id", "CustomerId"]))
    target_col = "Exited"
    input_cols = df.drop(columns=target_col).columns.tolist()
    X, y = split_features_target(df, input_cols, target_col)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    scaler = MinMaxScaler() if scaler_numeric else None
    X_train, X_test, scaler = scale_numeric_features(X_train, X_test, scaler)
    X_train, X_test, encoder = encode_categorical_features(X_train, X_test)

    return {
        "train_X": X_train,
        "train_y": y_train,
        "test_X": X_test,
        "test_y": y_test,
        "input_cols": input_cols,
        "target_col": target_col,
        "scaler": scaler,
        "encoder": encoder,
    }


def preprocess_new_data(new_df: pd.DataFrame, encoder: OneHotEncoder, scaler: MinMaxScaler = None) -> pd.DataFrame:
    """
    Preprocesses new data using the trained scaler and encoder.

    Args:
        new_df (pd.DataFrame): The new input data.
        encoder (OneHotEncoder): Trained encoder.
        scaler (MinMaxScaler): Trained scaler.

    Returns:
        pd.DataFrame: Processed new data.
    """
    X = with_surname_features(drop_columns(
        new_df.copy(), ["id", "CustomerId"]))

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    if scaler:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_cat = encoder.transform(X[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    X = pd.concat([X.drop(columns=categorical_cols), pd.DataFrame(
        X_cat, index=X.index, columns=encoded_cols)], axis=1)

    return X

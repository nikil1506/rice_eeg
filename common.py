import pandas as pd


def get_column_names(train_df):
    target = "main.disorder"
    features = [
        col
        for col in train_df.columns
        if col.startswith("AB.") or col.startswith("COH.")
    ]
    return target, features


def get_train_df():
    train_df = pd.read_csv("data/train.csv")
    target, features = get_column_names(train_df)
    return train_df[features + [target]]


def get_accuracy(df):
    accuracy = (df["y_true"] == df["y_pred"]).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")

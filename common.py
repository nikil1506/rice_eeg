import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def get_column_names(train_df):
    target = "main.disorder"
    features = [
        col
        for col in train_df.columns
        if col.startswith("AB.") or col.startswith("COH.")
    ]
    # features = [    # Schizophrenia (alpha PSD)
    # "AB.C.alpha.a.FP1",
    # "AB.C.alpha.b.FP2",
    # "AB.C.alpha.c.F7",
    # "AB.C.alpha.d.F3",
    # "AB.C.alpha.e.Fz",
    # "AB.C.alpha.f.F4",
    # "AB.C.alpha.g.F8",
    # "AB.C.alpha.h.T3",
    # "AB.C.alpha.i.C3",
    # "AB.C.alpha.j.Cz",
    # "AB.C.alpha.k.C4",
    # "AB.C.alpha.l.T4",
    # "AB.C.alpha.m.T5",
    # "AB.C.alpha.n.P3",
    # "AB.C.alpha.o.Pz",
    # "AB.C.alpha.p.P4",
    # "AB.C.alpha.q.T6",
    # "AB.C.alpha.r.O1",
    # "AB.C.alpha.s.O2",
    # Trauma and stress-related disorders (beta FC)
    # "COH.D.beta.a.FP1.b.FP2",
    # "COH.D.beta.c.F7.d.F3",
    # "COH.D.beta.e.Fz.f.F4",
    # "COH.D.beta.g.F8.h.T3",
    # "COH.D.beta.i.C3.j.Cz",
    # "COH.D.beta.k.C4.l.T4",
    # "COH.D.beta.m.T5.n.P3",
    # "COH.D.beta.o.Pz.p.P4",
    # "COH.D.beta.q.T6.r.O1",
    # Anxiety disorders (whole band PSD: all frequency bands)
    # "AB.A.delta.a.FP1",
    # "AB.A.delta.b.FP2",
    # "AB.B.theta.a.FP1",
    # "AB.B.theta.b.FP2",
    # "AB.C.alpha.a.FP1",
    # "AB.C.alpha.b.FP2",
    # "AB.D.beta.a.FP1",
    # "AB.D.beta.b.FP2",
    # "AB.E.highbeta.a.FP1",
    # "AB.E.highbeta.b.FP2",
    # "AB.F.gamma.a.FP1",
    # "AB.F.gamma.b.FP2",
    # Mood disorders (theta FC)
    # "COH.B.theta.a.FP1.b.FP2",
    # "COH.B.theta.c.F7.d.F3",
    # "COH.B.theta.e.Fz.f.F4",
    # "COH.B.theta.g.F8.h.T3",
    # "COH.B.theta.i.C3.j.Cz",
    # "COH.B.theta.k.C4.l.T4",
    # "COH.B.theta.m.T5.n.P3",
    # "COH.B.theta.o.Pz.p.P4",
    # "COH.B.theta.q.T6.r.O1",
    # Addictive disorders (theta PSD)
    # "AB.B.theta.a.FP1",
    # "AB.B.theta.b.FP2",
    # "AB.B.theta.c.F7",
    # "AB.B.theta.d.F3",
    # "AB.B.theta.e.Fz",
    # "AB.B.theta.f.F4",
    # "AB.B.theta.g.F8",
    # "AB.B.theta.h.T3",
    # "AB.B.theta.i.C3",
    # "AB.B.theta.j.Cz",
    # "AB.B.theta.k.C4",
    # "AB.B.theta.l.T4",
    # "AB.B.theta.m.T5",
    # "AB.B.theta.n.P3",
    # "AB.B.theta.o.Pz",
    # "AB.B.theta.p.P4",
    # "AB.B.theta.q.T6",
    # "AB.B.theta.r.O1",
    # "AB.B.theta.s.O2",
    # ]
    return target, features


def get_train_df():
    train_df = pd.read_csv("data/train.csv")
    target, features = get_column_names(train_df)
    train_df = train_df[features + [target]]

    if "sex" in train_df.columns:
        train_df["sex"] = train_df["sex"].map({"M": 0, "F": 1})
    return train_df


def under_sample_df(df):
    X = df.drop(columns=["main.disorder"])
    y = df["main.disorder"]
    undersample = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = undersample.fit_resample(X, y)
    df = X_resampled.copy()
    df["main.disorder"] = y_resampled
    return df


def over_sample_df(df):
    from imblearn.over_sampling import SMOTE

    X = df.drop(columns=["main.disorder"])
    y = df["main.disorder"]
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = X_resampled.copy()
    df_balanced["main.disorder"] = y_resampled
    return df_balanced


def get_accuracy(df):
    accuracy = (df["y_true"] == df["y_pred"]).mean() * 100
    print(f"Accuracy: {accuracy:.2f}%")


def print_class_distribution(y):
    print(pd.Series(y.numpy()).value_counts(normalize=True))

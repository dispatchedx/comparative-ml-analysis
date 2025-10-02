import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import numpy as np

# reading csv data
data = pd.read_csv("data/winequality-red.csv")
df = data
# labels
X = data[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
].values
# target
y = data["quality"].values


def train_svc_model(X, y):
    """
    Splits training-testing data to 75%-25%, trains a SVC model and prints its metrics
    """

    print(X.shape)  # (rows, features)
    print(y.shape)  # (rows,)
    # Split data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)

    # Train SVC
    svclassifier = SVC(kernel="linear")
    print("Training SVC model...")
    svclassifier.fit(X_train, y_train)

    # Evaluate
    y_predict = svclassifier.predict(X_test)
    print("Results:\n")
    print(metrics.classification_report(y_test, y_predict))


def remove_33_of_values(df, column):
    """
    Randomly sets 33% of the a column values in the DataFrame to NaN.

    """
    mask = df.sample(frac=0.33, random_state=1).index
    df.loc[mask, column] = np.nan
    print(f"Removed 33% of {column} values")


def fill_missing_with_average(df, column):
    average = df[column].mean()
    df[column] = df[column].fillna(average)

    print(f"33% of pH {column} replaced with average pH={average}")


print(
    "What to do?\n"
    "0. Correlation and feauture importances\n"
    "1. Part A (split dataset to training-test 75%-25% and predict quality)\n"
    "2. Part B (remove 33% of ph values and fill them with various ways)\n"
    "Any other key. Quit\n"
)

choice = input()
if choice == "0":
    # compute correlation of all features with 'quality'
    corr = df.corr()["quality"].sort_values(ascending=False)
    print(corr)
    X = df.drop(columns="quality").values
    y = df["quality"].values

    rf = RandomForestClassifier(random_state=1)
    rf.fit(X, y)
    # feature importances
    importances = pd.Series(rf.feature_importances_, index=df.drop(columns="quality").columns)
    print(importances.sort_values(ascending=False))


if choice == "1":
    print("---default SVC linear ---")
    train_svc_model(X, y)

    def train_better_model(X, y):
        # Split data
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)

        # Scale features (helps SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Try Random Forest
        print("\n--- Random Forest ---")
        rf = RandomForestClassifier(n_estimators=100, random_state=1, class_weight="balanced")
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        print(metrics.classification_report(y_test, y_pred_rf))

        # Try SVM with better parameters
        print("\n--- SVM (RBF kernel, balanced) ---")
        svm = SVC(kernel="rbf", class_weight="balanced", C=10)
        svm.fit(X_train_scaled, y_train)
        y_pred_svm = svm.predict(X_test_scaled)
        print(metrics.classification_report(y_test, y_pred_svm))

    X = df.drop(columns=["quality"]).values
    y = df["quality"]
    train_better_model(X, y)
    exit(1)
elif choice == "2":
    # remove_33_of_values(df, "alcohol")
    print(
        "What to do?\n"
        "1. Remove column alcohol\n"
        "2. Fill missing values with average alcohol\n"
        "3. Fill missing values with Logistic Regression\n"
        "4. Fill missing values with K-means cluster average\n"
        "5. Try other models\n"
        "Any other key. Quit\n"
    )
    choice = input()

    if choice == "1":
        remove_33_of_values(df, "alcohol")
        X = df.drop(columns=["alcohol", "quality"]).values
        y = df["quality"].values
        print("pH alcohol removed")
        train_svc_model(X, y)

    elif choice == "2":
        remove_33_of_values(df, "alcohol")
        fill_missing_with_average(df, "alcohol")
        X = df.drop(columns=["quality"]).values
        y = df["quality"].values
        train_svc_model(X, y)

    elif choice == "3":
        remove_33_of_values(df, "alcohol")
        X = df.drop(columns=["quality"]).values
        y = df["alcohol"].values

        has_alcohol = df["alcohol"].notna()
        missing_alcohol = df["alcohol"].isna()

        X_train_alcohol = df.loc[has_alcohol].drop(columns=["alcohol", "quality"]).values
        y_train_alcohol = df.loc[has_alcohol, "alcohol"].values

        X_predict_alcohol = df.loc[missing_alcohol].drop(columns=["quality", "alcohol"])

        regress = linear_model.LinearRegression()
        print("Training regression model..")
        # fit regression model
        regress.fit(X_train_alcohol, y_train_alcohol)
        print("Regression model finished training")
        # predict missing alcohol values
        print("Predicting missing alcohol values...")
        alcohol_predictions = regress.predict(X_predict_alcohol)

        # fill the missing values
        df.loc[missing_alcohol, "alcohol"] = alcohol_predictions
        print(f"Filled {missing_alcohol.sum()} missing alcohol values with regression predictions")

        # train quality classifier with complete data
        X = df.drop(columns=["quality"]).values
        y = df["quality"].values
        train_svc_model(X, y)

    elif choice == "4":
        remove_33_of_values(df, "alcohol")
        X_for_clustering = df.drop(columns=["alcohol", "quality"]).values

        kmeans = KMeans(n_clusters=6, random_state=1)
        kmeans.fit(X_for_clustering)

        df["cluster"] = kmeans.labels_

        has_alcohol = df["alcohol"].notna()
        cluster_alcohol_means = df.loc[has_alcohol].groupby("cluster")["alcohol"].mean()
        print("Average alcohol per cluster:")
        print(cluster_alcohol_means)

        missing_alcohol = df["alcohol"].isna()

        def fill_with_cluster_mean(row):
            if pd.isna(row["alcohol"]):
                return cluster_alcohol_means[row["cluster"]]
            return row["alcohol"]

        df["alcohol"] = df.apply(fill_with_cluster_mean, axis=1)
        print(f"Filled {missing_alcohol.sum()} missing alcohol values with cluster averages")

        # train quality classifier
        df.drop(columns=["cluster"], inplace=True)  # Remove helper column
        X = df.drop(columns=["quality"]).values
        y = df["quality"].values
        train_svc_model(X, y)
        # plt.scatter(data['quality'], data['pH'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
        # plt.show()

    else:
        exit(1)
else:
    exit(1)

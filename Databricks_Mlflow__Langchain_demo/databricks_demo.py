from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/claims_escalation_demo")

X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

with mlflow.start_run(run_name="claims_escalation_classifier_train_test"):

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("test_rows", len(X_test))

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1", f1)

    report = classification_report(y_test, preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv")

    cm = pd.DataFrame(
        confusion_matrix(y_test, preds),
        columns=["pred_normal", "pred_escalate"],
        index=["actual_normal", "actual_escalate"]
    )
    cm.to_csv("confusion_matrix.csv")
    mlflow.log_artifact("confusion_matrix.csv")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="claims_escalation_model",
        input_example=X_train[:3]
    )

    print("Accuracy:", acc)
    print("F1:", f1)
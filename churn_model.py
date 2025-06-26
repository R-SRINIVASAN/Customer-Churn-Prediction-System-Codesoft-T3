import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.progress import track
from rich import box
import time
import joblib

# 🎨 Setup Console
console = Console()

# 🧠 Typing Animation Intro
intro = "✨ Welcome to the AI-Powered Customer Churn Prediction System ✨"
with console.status("[bold green]Loading components...[/bold green]", spinner="earth"):
    for char in intro:
        console.print(char, end='', style="bold cyan")
        time.sleep(0.015)
    console.print("\n")

# 📄 Load Dataset
df = pd.read_csv("Churn_Modelling.csv")
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# 🔁 Encode Categorical Features
le_geo = LabelEncoder()
le_gen = LabelEncoder()
df["Geography"] = le_geo.fit_transform(df["Geography"])
df["Gender"] = le_gen.fit_transform(df["Gender"])

# 🔍 Define Features & Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# 🧪 Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔄 Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🧠 Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

console.rule("[bold cyan]🔍 Model Evaluation Phase")

# 📊 Accuracy Comparison Table
model_results = Table(title="🤖 Model Accuracy Comparison", box=box.DOUBLE)
model_results.add_column("Model Name", style="bold yellow", justify="center")
model_results.add_column("Accuracy (%)", style="bold green", justify="center")

best_model = None
best_accuracy = 0
best_name = ""

# 🔄 Train Models with Animation
for name in track(models.keys(), description="🔎 Training models..."):
    model = models[name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    model_results.add_row(name, f"{acc:.2f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

console.print(Align.center(model_results))

# 📋 Detailed Classification Report
y_pred_best = best_model.predict(X_test)
console.rule(f"[bold magenta]📊 Classification Report: {best_name}")

report = classification_report(y_test, y_pred_best, output_dict=True, target_names=["Not Churned", "Churned"])
report_table = Table(title="📋 Classification Metrics", box=box.SIMPLE_HEAVY)
report_table.add_column("Class", style="bold blue")
report_table.add_column("Precision", justify="center")
report_table.add_column("Recall", justify="center")
report_table.add_column("F1 Score", justify="center")

for cls, metrics in report.items():
    if cls not in ["accuracy", "macro avg", "weighted avg"]:
        report_table.add_row(cls, f"{metrics['precision']:.2f}", f"{metrics['recall']:.2f}", f"{metrics['f1-score']:.2f}")

console.print(Align.center(report_table))

# 📦 Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
console.rule("[bold red]🧾 Confusion Matrix")

cm_table = Table.grid(padding=1)
cm_table.add_column(justify="center")
cm_table.add_column(justify="center")
cm_table.add_column(justify="center")
cm_table.add_row("", "[bold red]Predicted: Churn[/bold red]", "[bold green]Predicted: Stay[/bold green]")
cm_table.add_row("[bold red]Actual: Churn[/bold red]", str(cm[1][1]), str(cm[1][0]))
cm_table.add_row("[bold green]Actual: Stay[/bold green]", str(cm[0][1]), str(cm[0][0]))
console.print(Align.center(Panel(cm_table, title="Confusion Matrix", subtitle="Churn vs Stay", style="red")))

# 💾 Save Artifacts
joblib.dump(best_model, "best_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_geo, "geo_encoder.pkl")
joblib.dump(le_gen, "gender_encoder.pkl")

# -----------------------------
# 🧪 Test Case Predictions
# -----------------------------
console.rule("[bold green]🧠 Predicting on Test Scenarios")

test_cases = [
    {"CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 30, "Tenure": 5, "Balance": 50000, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 60000},
    {"CreditScore": 700, "Geography": "Germany", "Gender": "Male", "Age": 45, "Tenure": 3, "Balance": 100000, "NumOfProducts": 2, "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 80000},
    {"CreditScore": 580, "Geography": "Spain", "Gender": "Female", "Age": 35, "Tenure": 8, "Balance": 70000, "NumOfProducts": 3, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000},
    {"CreditScore": 820, "Geography": "France", "Gender": "Male", "Age": 29, "Tenure": 1, "Balance": 30000, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 120000},
    {"CreditScore": 400, "Geography": "Germany", "Gender": "Female", "Age": 50, "Tenure": 9, "Balance": 0, "NumOfProducts": 1, "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 40000},
]

geo_enc = joblib.load("geo_encoder.pkl")
gen_enc = joblib.load("gender_encoder.pkl")
scaler = joblib.load("scaler.pkl")

predict_table = Table(title="🚀 AI Predictions on Sample Customers", box=box.ROUNDED)
predict_table.add_column("Case", style="bold cyan", justify="center")
predict_table.add_column("Geography", justify="center")
predict_table.add_column("Gender", justify="center")
predict_table.add_column("Age", justify="center")
predict_table.add_column("Prediction", style="bold magenta", justify="center")

for idx, test in enumerate(test_cases, 1):
    df_test = pd.DataFrame([{
        "CreditScore": test["CreditScore"],
        "Geography": geo_enc.transform([test["Geography"]])[0],
        "Gender": gen_enc.transform([test["Gender"]])[0],
        "Age": test["Age"],
        "Tenure": test["Tenure"],
        "Balance": test["Balance"],
        "NumOfProducts": test["NumOfProducts"],
        "HasCrCard": test["HasCrCard"],
        "IsActiveMember": test["IsActiveMember"],
        "EstimatedSalary": test["EstimatedSalary"]
    }])

    scaled_test = scaler.transform(df_test)
    prediction = best_model.predict(scaled_test)[0]
    result = "❌ Churn" if prediction == 1 else "✅ Stay"
    predict_table.add_row(f"{idx}", test["Geography"], test["Gender"], str(test["Age"]), result)

console.print(Align.center(predict_table))

# 🥳 Outro with Animation
console.rule("[bold blue]🎉 Churn Prediction Completed!")
outro = "🚀 You're now ready to prevent customer churn like a true data wizard!"
for char in outro:
    console.print(char, end='', style="bold green")
    time.sleep(0.015)
console.print("\n")

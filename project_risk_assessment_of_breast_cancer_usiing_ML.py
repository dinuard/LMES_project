import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance


# 1. Load and Clean Data
df = pd.read_csv("breast_cancer_bd.csv").apply(pd.to_numeric, errors='coerce')
df.fillna({"Bare Nuclei":df["Bare Nuclei"].median()}, inplace=True)
df.drop('Sample code number', axis=1, inplace=True)

# 2. Prepare Data
X = df.drop('Class', axis=1)
y = df['Class']

# Convert y to 0 and 1 (Benign: 0, Malignant: 1)
y = y.map({2: 0, 4: 1})

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 3. Exploratory Data Analysis (EDA)
print(df.describe())

#Visualize - Box Plot
plt.figure(figsize=(15, 8))
sns.boxplot(x='Class', y='Clump Thickness', data=df)
plt.title('Clump Thickness vs. Class')
plt.show()

#Visualize - Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#Visualize - Count Plot
plt.figure()
sns.countplot(x='Bare Nuclei', hue='Class', data=df)
plt.title('Distribution of Bare Nuclei by Class')
plt.show()

#Percentage of patients under risk of breast cancer
class_counts = df['Class'].value_counts()
cancer_percentage = (class_counts.get(4, 0) / len(df)) * 100  # Handle missing 4's
print(f"\nPercentage of patients with breast cancer: {cancer_percentage:.2f}%")

class_labels = {2: 'Benign', 4: 'Malignant'}
class_values = list(class_labels.keys())
class_names = list(class_labels.values())

#Visualize - Data Distribution _ Class wise
plt.figure()
plt.bar(class_values, class_counts.values)
plt.xticks(class_values, class_names)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Benign and Malignant Cases')
plt.show()

#Visualize - Risk Assessment
low_risk_threshold = df['Bare Nuclei'].quantile(0.33)
high_risk_threshold = df['Bare Nuclei'].quantile(0.66)
df['risk_category'] = pd.cut(df['Bare Nuclei'],
                            bins=[-1, low_risk_threshold, high_risk_threshold, df['Bare Nuclei'].max()],
                            labels=['Low Risk', 'Medium Risk', 'High Risk'], include_lowest=True)

plt.figure()
sns.countplot(x='risk_category', hue='Class', data=df, order=['Low Risk', 'Medium Risk', 'High Risk'])
plt.title('Distribution of Risk Categories by Class')
plt.show()

print("\nRisk Category Counts:\n", df['risk_category'].value_counts())

# 4. Feature Selection (RFE)
rfe_model = RandomForestClassifier(random_state=42)
rfe = RFE(rfe_model, n_features_to_select=5)
rfe.fit(X_train, y_train)
print("\nRFE Selected Features:", X.columns[rfe.support_])

# 5. Model Training and Evaluation
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(random_state=42, probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
results = {}

# Model Prediction
for name, model in models.items():
    print(f"\nTraining and Evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr, roc_auc = None, None, None

    results[name] = {"accuracy": accuracy, "report": report, "cm": cm, "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
#Model accuracy
    print(f"{name} Results:\nAccuracy: {accuracy}\nClassification Report:\n{report}")

#Model accuracy in visualization
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    if fpr is not None:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.show()

# 6. Model Comparison
print("\nModel Comparison:")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}")

# 7. Model Interpretation
if results:
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = models[best_model_name]  # Get the actual model object
    print(f"\nThe best model is: {best_model_name}")

#Permutatation Importance
    result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    X_original = X.copy()  # Create a copy of X for feature names

    # Get the feature names
    feature_names = X_original.columns  # Use X_original (DataFrame)

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T, vert=False, tick_labels=feature_names[sorted_idx])  # Use feature_names
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()

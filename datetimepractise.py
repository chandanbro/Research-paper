import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. LOAD & PREPARE DATA
# -----------------------------
data = pd.read_csv("combined_data.csv")
data.rename(columns={'ACC': 'activity', 'HR': 'hr', 'EDA': 'eda'}, inplace=True)


# -----------------------------
# 2. LABEL GENERATION (With Logic)
# -----------------------------
def create_label(row):
    score = 0
    if row['hr'] > 100: score += 1
    if row['hr'] < 60: score += 1
    if row['eda'] > 4.5: score += 1
    if row['eda'] < 7: score += 1
    if row['activity'] < 50: score += 1
    if row['activity'] > 88: score += 1

    if score >= 3:
        return 1 if np.random.rand() > 0.125 else 0
    else:
        return 0 if np.random.rand() > 0.068 else 1


data['label'] = data.apply(create_label, axis=1)

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------
# Add lag features to give the model "history"
data['hr_mean_5'] = data['hr'].rolling(5).mean().fillna(method='bfill')
data['eda_std_5'] = data['eda'].rolling(5).std().fillna(0)
data['activity_diff'] = data['activity'].diff().fillna(0)

X = data[['activity', 'hr', 'eda', 'hr_mean_5', 'eda_std_5', 'activity_diff']]
y = data['label']

# -----------------------------
# 4. TRAIN-TEST SPLIT & BALANCING
# -----------------------------
X_train, X_test, y_test_orig, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_test_orig)

# -----------------------------
# 5. FEDERATED TRAINING (STRATIFIED)
# -----------------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
client_models = []

for i, (train_idx, val_idx) in enumerate(skf.split(X_train_res, y_train_res)):
    X_c, y_c = X_train_res.iloc[train_idx], y_train_res.iloc[train_idx]

    # OPTIMIZED HYPERPARAMETERS
    model = RandomForestClassifier(
        n_estimators=300,  # Increased trees
        max_depth=15,  # Increased depth for better pattern recognition
        min_samples_leaf=2,
        class_weight='balanced',  # Automatic balancing instead of 1:4
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_c, y_c)
    client_models.append(model)
    print(f"Client {i + 1} Training Complete.")


# -----------------------------
# 6. OPTIMIZED PREDICTION FUNCTION
# -----------------------------
def federated_predict(models, X_input, threshold=0.5):
    all_probs = []
    for m in models:
        all_probs.append(m.predict_proba(X_input)[:, 1])

    avg_probs = np.mean(all_probs, axis=0)
    return (avg_probs >= threshold).astype(int)


# -----------------------------
# 7. EVALUATION & VISUALIZATION
# -----------------------------
# Setting threshold to 0.5 to fix the False Positive issue
FINAL_THRESHOLD = 0.5
y_pred = federated_predict(client_models, X_test, threshold=FINAL_THRESHOLD)

print("\n--- FINAL PERFORMANCE METRICS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# -----------------------------
# 14. PREDICT NEW VALUES
# -----------------------------

def predict_single_entry(activity, hr, eda, historical_df):
    """
    Calculates derived features and predicts for a single set of inputs.
    historical_df should be your 'data' dataframe to calculate rolling means.
    """
    # 1. Calculate the derived features based on your training logic
    hr_mean_5 = (historical_df['hr'].iloc[-4:].tolist() + [hr])
    hr_mean_5 = np.mean(hr_mean_5)

    eda_std_5 = (historical_df['eda'].iloc[-4:].tolist() + [eda])
    eda_std_5 = np.std(eda_std_5)

    activity_diff = activity - historical_df['activity'].iloc[-1]

    # 2. Create the feature vector (MUST match X.columns order)
    # Order: ['activity', 'hr', 'eda', 'hr_mean_5', 'eda_std_5', 'activity_diff']
    features = pd.DataFrame([[
        activity, hr, eda, hr_mean_5, eda_std_5, activity_diff
    ]], columns=['activity', 'hr', 'eda', 'hr_mean_5', 'eda_std_5', 'activity_diff'])

    # 3. Get prediction
    pred = federated_predict(client_models, features, threshold=FINAL_THRESHOLD)[0]

    status = "Depressive Pattern" if pred == 1 else "Normal"
    return status


# --- Example Usage ---
print("\n--- Live Prediction ---")
new_activity = float(input("Enter Activity score in terms of (40-120): "))
new_hr = float(input("Enter Avg Resting Heart Rate: "))
new_eda = float(input("Enter EDA score (2-10): "))

result = predict_single_entry(new_activity, new_hr, new_eda, data)
heko = result
print(f"Prediction: {result}")
if heko == 'Normal':
    print( "You are stable. Keep maintaining your healthy routine.")
else:
    print("Possible depressive pattern detected. ")
    print( "Consider rest, activity, and talking to someone. ")
    print("Seek professional help if it continues.")
print('This is not professional advice, AI can be wrong. So please consult to Doctor')





# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Depressive'],
            yticklabels=['Normal', 'Depressive'])
plt.title(f'Optimized Confusion Matrix (Threshold: {FINAL_THRESHOLD})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
# -----------------------------
# 10. CLASSIFICATION REPORT BAR GRAPH
# -----------------------------
from sklearn.metrics import precision_recall_fscore_support


def plot_classification_report(y_true, y_pred):
    # Extract metrics
    metrics = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = metrics

    # Data setup
    classes = ['Normal (0)', 'Depressive (1)']
    x = np.arange(len(classes))
    width = 0.2  # width of bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (0.0 to 1.0)')
    ax.set_title('Classification Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)  # Give space for the legend
    ax.legend(loc='upper right')

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.show()


# Call the function
plot_classification_report(y_test, y_pred)
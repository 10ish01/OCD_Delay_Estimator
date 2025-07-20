#========== IMPORTS ========== 
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from collections import Counter
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostRegressor, Pool

# LOAD & COMBINE LATEST DATASETS (The latest 5 datasets are taken and combined, rest are ignored)
folder = 'dataset'
files = sorted(glob.glob(f'{folder}/dataset*.csv'), key=os.path.getmtime)[-5:]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# PREPROCESSING & FEATURE ENGINEERING 
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df.rename(columns={
    'sample_received_time': 'sample_arrival_time',
    'department_scan': 'department_arrival_time',
    'tech_saved': 'tech_saved_time',
    'lab_delay_hours': 'delay'
}, inplace=True)

# Convert time columns
for col in ['sample_arrival_time', 'department_arrival_time', 'tech_saved_time']:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Fill missing department_arrival_time
mask_missing_dept = df['department_arrival_time'].isna()
df.loc[mask_missing_dept, 'department_arrival_time'] = df.loc[mask_missing_dept, 'sample_arrival_time'] + timedelta(minutes=30)

# Dropping rows with missing tech_saved_time
initial_len = len(df)
df = df[~df['tech_saved_time'].isna()]
dropped_rows = initial_len - len(df)

# Filling other missing values where possible
df['lab_tat'] = df['lab_tat'].fillna(df['lab_tat'].mode()[0])
df['test_name'] = df['test_name'].fillna("UNKNOWN")
df['barcode'] = df['barcode'].fillna("MISSING")

df['day'] = df['sample_arrival_time'].dt.dayofweek

#Binning the samples based on arrival time
def time_to_bin(t):
    hour = t.hour
    if hour < 7: return 0
    elif hour < 9: return 1
    elif hour < 11: return 2
    elif hour < 13: return 3
    elif hour < 16: return 4
    elif hour < 20: return 5
    else: return 6

df['time_bin'] = df['sample_arrival_time'].apply(time_to_bin)

# Logic for cumulative delays 
df = df.sort_values('sample_arrival_time')
df['date_only'] = df['sample_arrival_time'].dt.date
df['delay_flag'] = (df['delay'] > 0).astype(int)
df['cumulative_delays'] = (
    df.groupby('date_only')['delay_flag']
    .cumsum() - df['delay_flag']
)
 
#Logic for calculation of current processing tets
df['dept_date'] = df['department_arrival_time'].dt.date

df = df.sort_values(['dept_date', 'department_arrival_time']).reset_index(drop=True)

df['current_processing'] = 0

for day in df['dept_date'].unique():
    day_df = df[df['dept_date'] == day]
    dept_arr = day_df['department_arrival_time'].values
    tech_saved = day_df['tech_saved_time'].values

    result = np.zeros(len(day_df), dtype=int)

    for i in range(len(day_df)):
        result[i] = np.sum((dept_arr[:i] < dept_arr[i]) & (tech_saved[:i] > dept_arr[i]))

    df.loc[day_df.index, 'current_processing'] = result
    
#Other features
df['test_time'] = (df['tech_saved_time'] - df['department_arrival_time']).dt.total_seconds()/60
df = df[df['test_time'].between(df['test_time'].quantile(0.00), df['test_time'].quantile(0.90))]
df['avg_test_time'] = df.groupby('test_name')['test_time'].transform('mean')/60
df['tests_in_sample'] = df.groupby('barcode')['barcode'].transform('count')

df['delay_ratio'] = df.groupby('test_name')['delay'].transform(lambda x: (x > 0).mean())
df['avg_test_delay'] = df.groupby('test_name')['delay'].transform(lambda x: x[x > 0].mean())

# ADD BREAKDOWN INFO
break_df = pd.read_csv("data/breakdowns.csv")
break_df['breakdown_datetime'] = pd.to_datetime(break_df['breakdown_datetime'],dayfirst=True, errors='coerce')
break_df['timestamp'] = pd.to_datetime(break_df['timestamp'],dayfirst=True, errors='coerce')
break_df['date'] = break_df['breakdown_datetime'].dt.date

df['sample_date'] = df['sample_arrival_time'].dt.date

agg_breaks = break_df.groupby('date')['duration_mins'].sum().reset_index()
agg_breaks['breakdown_flag'] = 1

df = df.merge(agg_breaks, left_on='sample_date', right_on='date', how='left')
df['breakdown_flag'] = df['breakdown_flag'].fillna(0).astype(int)
df['breakdown_duration'] = df['duration_mins'].fillna(0)

def bin_breakdown(dur):
    if dur == 0: return 0
    elif dur <= 30: return 1
    elif dur <= 60: return 2
    elif dur <= 120: return 3
    else: return 4

df['breakdown_class'] = df['breakdown_duration'].apply(bin_breakdown)
nonzero_breakdowns = (df['breakdown_flag'] != 0).sum()

#Dropping unnecessary columns
df.drop(columns=['date_only', 'sample_date', 'date', 'duration_mins', 'dept_rank', 'tech_rank'], inplace=True, errors='ignore')

filtered = df[(df['delay'] > 0) & (df['breakdown_flag'] == 1)]
avg_delay_breakdown_map = filtered.groupby('test_name')['delay'].mean()
df['avg_delay_when_breakdown'] = df['test_name'].map(avg_delay_breakdown_map).fillna(0)

df.drop(columns=['barcode', 'lab_time', 'breakdown_duration'], inplace=True, errors='ignore')

df = df[df['lab_tat'].isin([3, 4])]
df.dropna(inplace=True)

df.drop(columns=[
    'barcode', 'lab_time',
    'sample_arrival_time', 'department_arrival_time', 'tech_saved_time',
    'test_time', 'date_only', 'sample_date', 'date',
    'dept_rank', 'tech_rank','dept_date','tech_date','delay_flag'], inplace=True, errors='ignore')

#Saving the final data that will be used to train the model
df.to_csv("data/train_data.csv", index=False)

# Saving information about all the tests as a separate file
test_metrics = df.groupby('test_name').agg({
    'avg_test_time': 'first',
    'lab_tat': 'first',
    'delay_ratio': 'first',
    'avg_test_delay': 'first',
    'avg_delay_when_breakdown': 'first'
}).reset_index()
test_metrics.to_csv("data/test_data.csv", index=False)

# MODEL TRAINING BEGINS HERE
df = pd.read_csv("data/train_data.csv")
df = df.dropna(subset=['delay'])

X = df.drop(columns=['delay','test_name'])
y = df['delay']

X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.astype(str).str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

numeric_cols = X.select_dtypes(include=['number']).columns
non_numeric = X.drop(columns=numeric_cols)

if len(numeric_cols) > 0:
    X_numeric = pd.DataFrame(
        SimpleImputer(strategy='mean').fit_transform(X[numeric_cols]),
        columns=numeric_cols
    )
else:
    print("No numeric columns found.")
    X_numeric = pd.DataFrame()

X = pd.concat([X_numeric, non_numeric.reset_index(drop=True)], axis=1)

y_class = (y > 0).astype(int)
X_train, X_val, y_train_class, y_val_class, y_train_reg, y_val_reg = train_test_split(
    X, y_class, y, test_size=0.2, random_state=42
)



counter = Counter(y_train_class)
minority_class_size = min(counter.values())
k_neighbors = max(1, minority_class_size - 1)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_bal, y_train_class_bal = smote.fit_resample(X_train, y_train_class)

clf = XGBClassifier(scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train_bal, y_train_class_bal)

os.makedirs('model', exist_ok=True)
joblib.dump(clf, 'model/classifier.pkl')

y_probs = clf.predict_proba(X_val)[:, 1]
prec, rec, thresh = precision_recall_curve(y_val_class, y_probs)
f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
best_idx = np.argmax(f1)
best_thresh = thresh[best_idx]
preds = (y_probs >= best_thresh).astype(int)


cm = confusion_matrix(y_val_class, preds)
os.makedirs('assets', exist_ok=True)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('assets/confusion_matrix.png')
plt.close()

report = classification_report(y_val_class, preds, digits=4)


os.makedirs("model", exist_ok=True)
with open("model/model_metrics.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report + "\n")


mask_train = y_train_class == 1
X_train_reg = X_train[mask_train]
y_train_reg = y_train_reg[mask_train]

mask_val = (preds == 1) & (y_val_reg > 0)
X_val_reg = X_val[mask_val]
y_val_reg = y_val_reg[mask_val]

if not X_val_reg.empty:
    y_train_log = np.log1p(y_train_reg)
    y_val_log = np.log1p(y_val_reg)
    weights = 1 + 4 / (1 + np.exp(-1.5 * (y_train_reg - 2)))

    train_pool = Pool(X_train_reg, y_train_log, weight=weights)
    val_pool = Pool(X_val_reg, y_val_log)

    cat_model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function='Quantile:alpha=0.7',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    cat_model.fit(train_pool, eval_set=val_pool)
    joblib.dump(cat_model, 'model/regressor.pkl')

    y_pred_log = cat_model.predict(X_val_reg)
    y_pred_actual = np.expm1(y_pred_log)
    y_pred_actual = np.clip(y_pred_actual, 0, None)

    with open("model/model_metrics.txt", "a") as f:
        f.write ("\nRegression Report:\n")
        f.write(f"MAE:   {mean_absolute_error(y_val_reg, y_pred_actual):.3f}\n")
        f.write(f"RMSE:  {np.sqrt(mean_squared_error(y_val_reg, y_pred_actual)):.3f}\n")
        f.write(f"R2:    {r2_score(y_val_reg, y_pred_actual):.3f}\n")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_val_reg, y_pred_actual, alpha=0.5, label='Predicted vs Actual')  
    plt.plot([0, 10], [0, 10], 'r--', label='Ideal Fit (y=x)')

    plt.xlabel("Actual Delay")
    plt.ylabel("Predicted Delay")
    plt.title("CatBoost: Actual vs Predicted")

    limit = max(y_val_reg.max(), y_pred_actual.max()) * 1.2  
    plt.xlim(0, limit)
    plt.ylim(0, limit)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    

    plt.savefig("assets/actual_vs_predicted_delay.png")
    plt.close()

    explainer = shap.Explainer(cat_model)
    shap_values = explainer(X_val_reg)
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig("assets/shap_plot.png", bbox_inches='tight')
    plt.close()
else:
    print("No delayed validation samples predicted. Regression skipped.")


print(f"\nRows dropped due to missing tech_saved_time: {dropped_rows}")
print(f"Rows with delay: {df[df['delay'] > 0].shape[0]}")
print(f"Rows with non-zero breakdown_flag: {nonzero_breakdowns}")
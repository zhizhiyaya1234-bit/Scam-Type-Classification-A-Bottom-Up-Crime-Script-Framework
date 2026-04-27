import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. 读取数据
# =========================
df = pd.read_csv("/Users/wangziyi/Downloads/model_training_ready.csv")
df.columns = [str(c).strip() for c in df.columns]

feature_cols = [
    "contact_primary",
    "trust_primary",
    "manipulation_primary",
    "operation_primary",
    "extraction_primary",
    "aftermath_primary",
    "psychological_vulnerability",
    "compliance_driver"
]
target_col = "final_type"

# 检查字段
missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少字段: {missing_cols}")

# 缺失统一
for col in feature_cols:
    df[col] = (
        df[col]
        .fillna("未提及")
        .astype(str)
        .str.strip()
        .replace({
            "": "未提及",
            "无": "未提及",
            "未使用": "未提及"
        })
    )

df = df[df[target_col].notna()].copy()
df[target_col] = df[target_col].astype(str).str.strip()

X = df[feature_cols]
y = df[target_col]

print("样本量:", len(df))
print("类别分布:")
print(y.value_counts())

# =========================
# 2. 预处理
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)
    ]
)

# =========================
# 3. 定义模型
# =========================
dt_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced"
    ))
])

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

# =========================
# 4. 5-fold CV
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro"
}

dt_scores = cross_validate(
    dt_model, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

rf_scores = cross_validate(
    rf_model, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

# =========================
# 5. 输出结果
# =========================
def print_cv_result(name, scores):
    print("\n" + "="*60)
    print(name)
    print("="*60)
    for i, (acc, f1) in enumerate(zip(scores["test_accuracy"], scores["test_macro_f1"]), start=1):
        print(f"Fold {i}: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    print("-"*60)
    print(f"Mean Accuracy = {np.mean(scores['test_accuracy']):.4f} ± {np.std(scores['test_accuracy']):.4f}")
    print(f"Mean Macro F1 = {np.mean(scores['test_macro_f1']):.4f} ± {np.std(scores['test_macro_f1']):.4f}")

print_cv_result("Decision Tree - 5 Fold CV", dt_scores)
print_cv_result("Random Forest - 5 Fold CV", rf_scores)

# =========================
# 6. 保存结果
# =========================
cv_result_df = pd.DataFrame({
    "fold": [1, 2, 3, 4, 5],
    "dt_accuracy": dt_scores["test_accuracy"],
    "dt_macro_f1": dt_scores["test_macro_f1"],
    "rf_accuracy": rf_scores["test_accuracy"],
    "rf_macro_f1": rf_scores["test_macro_f1"],
})

summary_row = pd.DataFrame({
    "fold": ["mean"],
    "dt_accuracy": [np.mean(dt_scores["test_accuracy"])],
    "dt_macro_f1": [np.mean(dt_scores["test_macro_f1"])],
    "rf_accuracy": [np.mean(rf_scores["test_accuracy"])],
    "rf_macro_f1": [np.mean(rf_scores["test_macro_f1"])],
})

cv_result_df = pd.concat([cv_result_df, summary_row], ignore_index=True)
cv_result_df.to_excel("/Users/wangziyi/Downloads/cv_results_dt_rf.xlsx", index=False)
print("\n已保存: cv_results_dt_rf.xlsx")
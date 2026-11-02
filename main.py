from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================== КОНФИГ ==============================
DATA_PATH = Path(r"C:\Users\sanda\OneDrive\Рабочий стол\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv")
REPORTS_DIR = Path(r"C:\Users\sanda\OneDrive\Рабочий стол\reports")
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPORTS_DIR / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Бизнес-допущения (легко менять под конкретную компанию)
CONTACT_COST = 50.0        # стоимость одного контакта/скидки
RETENTION_PROFIT = 500.0   # прибыль, если клиент остался
SUCCESS_RATE = 0.30        # вероятность, что кампания сработает
BUSINESS_THRESHOLD = CONTACT_COST / (SUCCESS_RATE * RETENTION_PROFIT)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


# ============================== УТИЛИТЫ ==============================
def log(msg: str) -> None:
    print(f"[main] {msg}")


def ensure_dirs() -> None:
    for d in (REPORTS_DIR, FIG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  сохранён график: {path}")


# ============================== ДАННЫЕ ==============================
def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл с данными: {path}\n"
            f"Скачайте датасет с Kaggle и положите его по этому пути."
        )
    return pd.read_csv(path)


def prepare_target_and_features(df: pd.DataFrame):
    """
    Возвращает:
      X — признаки (без customerID и Churn);
      y — бинарный таргет (1 = ушёл);
      df_eda — копия с колонкой Churn01 для графиков.
    """
    df = df.copy()

    # 1) TotalCharges хранится как object, в нём есть скрытые пропуски " ".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2) Целевая переменная
    y = (df["Churn"] == "Yes").astype(int)

    # 3) Признаки
    X = df.drop(columns=["customerID", "Churn"])

    # 4) Для EDA — та же таблица, но с числовым таргетом
    df_eda = df.copy()
    df_eda["Churn01"] = y

    return X, y, df_eda


# ============================== EDA ==============================
def run_eda(df_eda: pd.DataFrame) -> None:
    log("EDA: строим 4 базовых графика")

    # 1) Отток по типу контракта
    churn_rate = (
        df_eda.groupby("Contract")["Churn01"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax, palette="viridis")
    ax.set_title("Доля оттока по типу контракта")
    ax.set_ylabel("Доля ушедших")
    ax.set_xlabel("Тип контракта")
    for i, v in enumerate(churn_rate.values):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    save_fig(fig, "eda_churn_by_contract.png")

    # 2) Распределение tenure по оттоку
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df_eda, x="tenure", hue="Churn01", bins=30,
        kde=True, alpha=0.55, ax=ax,
    )
    ax.set_title("Распределение tenure (срок обслуживания) по оттоку")
    ax.set_xlabel("tenure, месяцев")
    save_fig(fig, "eda_tenure_by_churn.png")

    # 3) MonthlyCharges vs отток
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_eda, x="Churn01", y="MonthlyCharges", ax=ax)
    ax.set_title("Ежемесячный платёж по факту оттока")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Остался (0)", "Ушёл (1)"])
    ax.set_xlabel("")
    save_fig(fig, "eda_monthly_by_churn.png")

    # 4) TotalCharges vs tenure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_eda, x="tenure", y="TotalCharges",
        hue="Churn01", alpha=0.4, ax=ax,
    )
    ax.set_title("TotalCharges vs tenure (цвет — факт оттока)")
    save_fig(fig, "eda_totalcharges_vs_tenure.png")


# ============================== МОДЕЛИ ==============================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    log(f"  числовые признаки : {num_cols}")
    log(f"  категориальные    : {len(cat_cols)} шт.")

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def build_models() -> dict:
    return {
        "Dummy (prior)": DummyClassifier(
            strategy="prior", random_state=RANDOM_STATE
        ),
        "LogReg": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "DecisionTree (depth=5)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest (200)": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


# ============================== МЕТРИКИ ==============================
def metrics_at(y_true, y_proba, threshold: float) -> dict:
    pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float((pred == y_true).mean()), 4),
        "contacted": int(pred.sum()),
        "n_total": int(len(y_true)),
    }


def campaign_profit(y_true, y_proba, threshold: float) -> float:
    """Ожидаемая прибыль кампании на офлайн-разметке при данном пороге."""
    pred = (y_proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    contacted = int(pred.sum())
    return tp * SUCCESS_RATE * RETENTION_PROFIT - contacted * CONTACT_COST


def profit_curve(y_true, y_proba) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        contacted = int(pred.sum())
        profit = tp * SUCCESS_RATE * RETENTION_PROFIT - contacted * CONTACT_COST
        rows.append({
            "threshold": float(t),
            "contacted": contacted,
            "true_positives": tp,
            "expected_profit": float(profit),
        })
    return pd.DataFrame(rows)


# ============================== ГРАФИКИ МОДЕЛИ ==============================
def plot_roc_pr(y_true, y_proba, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC-кривая — {model_name}")
    ax.legend(loc="lower right")
    save_fig(fig, "roc_curve.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {model_name}")
    ax.legend(loc="lower left")
    save_fig(fig, "pr_curve.png")


def plot_confusion(y_true, y_proba, threshold: float, model_name: str) -> None:
    pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Остался", "Ушёл"],
        yticklabels=["Остался", "Ушёл"],
    )
    ax.set_xlabel(f"Predicted (порог={threshold:.2f})")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    save_fig(fig, "confusion_matrix.png")


def plot_profit(curve: pd.DataFrame, t_opt: float, t_business: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["threshold"], curve["expected_profit"], color="C0")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, label="No campaign")
    ax.axvline(t_business, color="C1", linestyle=":", label=f"Бизнес-порог = {t_business:.3f}")
    ax.axvline(t_opt, color="C2", linestyle="-.", label=f"Оптимум по OOF = {t_opt:.2f}")
    ax.set_xlabel("Порог вероятности оттока")
    ax.set_ylabel("Ожидаемая прибыль кампании")
    ax.set_title("Прибыль retention-кампании vs порог контакта")
    ax.legend()
    save_fig(fig, "profit_vs_threshold.png")


def plot_feature_importance(best_pipe: Pipeline, model_name: str) -> None:
    preprocessor = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["model"]

    try:
        names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    except Exception:
        return

    if hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        title = f"Коэффициенты {model_name} (топ-15 по |coef|)"
        xlabel = "Коэффициент"
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
        title = f"Важность признаков — {model_name} (топ-15)"
        xlabel = "Важность"
    else:
        return

    order = np.argsort(np.abs(values))[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=values[order],
        y=[names[i] for i in order],
        ax=ax, palette="coolwarm",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save_fig(fig, "feature_importance.png")


# ============================== ПРЕДСКАЗАНИЕ ==============================
def predict_client(pipeline: Pipeline, client: dict, threshold: float) -> dict:
    """
    client — dict с колонками как у X (без customerID и Churn).
    Возвращает вероятность оттока и рекомендацию по кампании.
    """
    row = pd.DataFrame([client])
    proba = float(pipeline.predict_proba(row)[0, 1])
    decision = "контактить" if proba >= threshold else "не контактить"
    expected = proba * SUCCESS_RATE * RETENTION_PROFIT - CONTACT_COST
    return {
        "churn_probability": round(proba, 4),
        "threshold": round(float(threshold), 4),
        "decision": decision,
        "expected_profit_if_contact": round(expected, 2),
    }


# ============================== MAIN ==============================
def main() -> None:
    ensure_dirs()

    # 1. Данные
    log(f"Загружаем данные из {DATA_PATH}")
    df = load_raw(DATA_PATH)
    X, y, df_eda = prepare_target_and_features(df)
    log(f"  строк: {len(df)}, колонок-признаков: {X.shape[1]}")
    log(f"  доля оттока: {y.mean():.2%}")

    # 2. EDA
    run_eda(df_eda)

    # 3. Train / Test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log(f"train: {len(X_tr)} | test: {len(X_te)}")

    # 4. Препроцессор и модели
    preprocessor = build_preprocessor(X_tr)
    models = build_models()

    # 5. 5-fold CV, OOF-предсказания, выбор лучшей модели по PR-AUC
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = {}
    oof_probas = {}

    for name, model in models.items():
        log(f"CV {CV_FOLDS}-fold: {name}")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        oof = cross_val_predict(
            pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        oof_probas[name] = oof
        cv_rows[name] = {
            "cv_roc_auc": round(float(roc_auc_score(y_tr, oof)), 4),
            "cv_pr_auc": round(float(average_precision_score(y_tr, oof)), 4),
        }

    cv_df = (
        pd.DataFrame(cv_rows).T
        .sort_values("cv_pr_auc", ascending=False)
    )
    print("\n=== CV-результаты (на train) ===")
    print(cv_df.to_string())
    cv_df.to_csv(REPORTS_DIR / "cv_results.csv")

    best_name = cv_df.index[0]
    log(f"Лучшая модель по CV PR-AUC: {best_name}")

    # 6. Выбор бизнес-порога
    y_oof = oof_probas[best_name]
    curve = profit_curve(y_tr, y_oof)
    best_row = curve.loc[curve["expected_profit"].idxmax()]
    t_opt = float(best_row["threshold"])
    log(
        f"Эмпирический оптимум порога по OOF: {t_opt:.2f}, "
        f"ожидаемая прибыль (OOF) = {best_row['expected_profit']:.0f}"
    )
    log(
        f"Аналитический бизнес-порог = C/(s*L) = "
        f"{CONTACT_COST}/({SUCCESS_RATE}*{RETENTION_PROFIT}) = {BUSINESS_THRESHOLD:.3f}"
    )
    plot_profit(curve, t_opt, BUSINESS_THRESHOLD)

    # 7. Финальное обучение лучшей модели на train
    best_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[best_name]),
    ])
    best_pipe.fit(X_tr, y_tr)
    proba_te = best_pipe.predict_proba(X_te)[:, 1]

    # 8. Оценка на test при трёх порогах
    test_metrics = {
        "at_0.5": metrics_at(y_te, proba_te, 0.5),
        "at_business_threshold": metrics_at(y_te, proba_te, BUSINESS_THRESHOLD),
        "at_opt_threshold": metrics_at(y_te, proba_te, t_opt),
    }
    print("\n=== Метрики на test ===")
    print(pd.DataFrame(test_metrics).T.to_string())
    pd.DataFrame(test_metrics).T.to_csv(REPORTS_DIR / "test_metrics.csv")

    # 9. Бизнес-сравнение на test
    n_te = len(y_te)
    profit_none = 0.0
    profit_all = (
        int(y_te.sum()) * SUCCESS_RATE * RETENTION_PROFIT - n_te * CONTACT_COST
    )
    profit_model_opt = campaign_profit(y_te, proba_te, t_opt)
    profit_model_business = campaign_profit(y_te, proba_te, BUSINESS_THRESHOLD)

    business_summary = {
        "assumptions": {
            "contact_cost": CONTACT_COST,
            "retention_profit": RETENTION_PROFIT,
            "success_rate": SUCCESS_RATE,
        },
        "analytical_business_threshold": round(BUSINESS_THRESHOLD, 4),
        "empirical_best_threshold_on_oof": round(t_opt, 4),
        "test_profit": {
            "no_campaign": round(profit_none, 2),
            "contact_everyone": round(profit_all, 2),
            f"model@{BUSINESS_THRESHOLD:.3f}": round(profit_model_business, 2),
            f"model@{t_opt:.2f}": round(profit_model_opt, 2),
        },
        "best_model": best_name,
    }
    with open(REPORTS_DIR / "business_summary.json", "w", encoding="utf-8") as f:
        json.dump(business_summary, f, ensure_ascii=False, indent=2)

    print("\n=== Бизнес-сравнение на test ===")
    for k, v in business_summary["test_profit"].items():
        print(f"  {k:>24}: {v:>12.2f}")

    # 10. Диагностические графики
    plot_roc_pr(y_te, proba_te, best_name)
    plot_confusion(y_te, proba_te, t_opt, best_name)
    plot_feature_importance(best_pipe, best_name)

    # 11. Classification report
    pred_te = (proba_te >= t_opt).astype(int)
    report = classification_report(
        y_te, pred_te,
        target_names=["Остался", "Ушёл"],
        digits=4, zero_division=0,
    )
    print("\n=== Classification report (test, t_opt) ===")
    print(report)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # 12. Сохранение модели и сводных метрик
    joblib.dump(best_pipe, MODELS_DIR / "churn_pipeline.joblib")
    log(f"Модель сохранена: {MODELS_DIR / 'churn_pipeline.joblib'}")

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_results": cv_rows,
                "best_model": best_name,
                "test_metrics": test_metrics,
                "business_summary": business_summary,
            },
            f, ensure_ascii=False, indent=2,
        )

    # 13. Демонстрация predict_client
    log("Демонстрация predict_client для одного клиента из test")
    sample_client = X_te.iloc[0].to_dict()
    result = predict_client(best_pipe, sample_client, t_opt)
    print("\n=== Пример предсказания ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    log("Готово. Смотрите reports/.")


if __name__ == "__main__":
    main()
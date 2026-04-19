import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_squared_error,
                             mean_absolute_error, r2_score)

def main():
    print("Loading prepared data...")
    df = pd.read_csv("cleaned_data.csv")
    
    # Target columns
    target_cls = "Good_Investment"
    target_reg = "Future_Price_5Y"
    
    # Load the official feature columns used for modeling
    try:
        feature_columns = joblib.load("models/feature_columns.pkl")
    except:
        feature_columns = [c for c in df.columns if c not in ["ID", target_cls, target_reg]]
    
    X = df[feature_columns]
    y_cls = df[target_cls]
    y_reg = df[target_reg]
    
    print("Splitting dataset...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # Set up MLFlow experiment
    mlflow.set_experiment("RealEstate_Investment_Advisor")
    
    # ==========================
    # CLASSIFICATION MODELS
    # ==========================
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=100, n_jobs=-1, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(n_estimators=50, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_cls_model = None
    best_cls_name = ""
    best_cls_score = -1
    best_cls_run_id = None
    
    os.makedirs("models", exist_ok=True)
    
    print("Training Classification Models...")
    for name, clf in classifiers.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"Training {name}...")
            clf.fit(X_train_c, y_train_c)
            y_pred = clf.predict(X_test_c)
            
            acc = accuracy_score(y_test_c, y_pred)
            prec = precision_score(y_test_c, y_pred, zero_division=0)
            rec = recall_score(y_test_c, y_pred, zero_division=0)
            f1 = f1_score(y_test_c, y_pred, zero_division=0)
            
            try:
                y_prob = clf.predict_proba(X_test_c)[:, 1]
                roc_auc = roc_auc_score(y_test_c, y_prob)
            except:
                roc_auc = 0.0
            
            # Log params & metrics
            mlflow.log_param("model_type", name)
            # Log hyperparameters based on model type
            if "Random Forest" in name:
                mlflow.log_param("n_estimators", 100)
            elif "Logistic" in name:
                mlflow.log_param("max_iter", 1000)
                
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Precision", prec)
            mlflow.log_metric("Recall", rec)
            mlflow.log_metric("F1_Score", f1)
            mlflow.log_metric("ROC_AUC", roc_auc)
            
            # Log model
            mlflow.sklearn.log_model(clf, "model")
            
            # Generate and log Confusion Matrix
            cm = confusion_matrix(y_test_c, y_pred)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{name} - Confusion Matrix")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            cm_path = f"models/{name.replace(' ', '_')}_cm.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            
            if f1 > best_cls_score:
                best_cls_score = f1
                best_cls_model = clf
                best_cls_name = name
                best_cls_run_id = run.info.run_id
                
    # Save best classifier
    print(f"Best Classifier: {best_cls_name} (F1: {best_cls_score:.4f})")
    joblib.dump(best_cls_model, "models/best_classifier.pkl")
    # Register the model
    try:
        model_uri = f"runs:/{best_cls_run_id}/model"
        mlflow.register_model(model_uri, "Best_RealEstate_Classifier")
    except Exception as e:
        print("Could not register classifier:", e)

    # ==========================
    # REGRESSION MODELS
    # ==========================
    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=50, n_jobs=-1, objective='reg:squarederror', random_state=42)
    }
    
    best_reg_model = None
    best_reg_name = ""
    best_reg_score = float('inf') # MAE or RMSE
    best_reg_run_id = None
    
    print("Training Regression Models...")
    for name, reg in regressors.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"Training {name}...")
            reg.fit(X_train_r, y_train_r)
            y_pred = reg.predict(X_test_r)
            
            rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
            mae = mean_absolute_error(y_test_r, y_pred)
            r2 = r2_score(y_test_r, y_pred)
            
            # Log params & metrics
            mlflow.log_param("model_type", name)
            if "Random Forest" in name:
                mlflow.log_param("n_estimators", 100)
                
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            
            # Log model
            mlflow.sklearn.log_model(reg, "model")
            
            # Smaller is better for RMSE
            if rmse < best_reg_score:
                best_reg_score = rmse
                best_reg_model = reg
                best_reg_name = name
                best_reg_run_id = run.info.run_id
                
    # Save best regressor
    print(f"Best Regressor: {best_reg_name} (RMSE: {best_reg_score:.4f})")
    joblib.dump(best_reg_model, "models/best_regressor.pkl")
    
    try:
        model_uri = f"runs:/{best_reg_run_id}/model"
        mlflow.register_model(model_uri, "Best_RealEstate_Regressor")
    except Exception as e:
        print("Could not register regressor:", e)

    print("Model training complete.")

if __name__ == "__main__":
    main()

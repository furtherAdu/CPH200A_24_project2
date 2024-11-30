
import argparse
import json
from csv import DictReader
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import shap

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--plco_data_path",
        default="/Users/rbhalerao/Desktop/project1_modified/plco/lung_prsn.csv",
        help="Location of PLCO csv",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.0006,
        type=float,
        help="Learning rate to use for SGD",
    )

    parser.add_argument(
        "--regularization_lambda",
        default=0,
        type=float,
        help="Weight to use for L2 regularization",
    )

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch_size to use for SGD"
    )

    parser.add_argument(
        "--num_epochs",
        default=200,
        type=int,
        help="number of epochs to use for training"
    )

    parser.add_argument(
        "--results_path",
        default="results.json",
        help="Where to save results"
    )

    parser.add_argument(
        '--features', 
        default='features.json',
        help='JSON defining features to use')
    return parser

def load_data(args: argparse.Namespace) -> ([list, list, list]):
    '''
    Load PLCO data from csv file and split into train validation and testing sets. 
    Data loaded into a list of dictionaries where each dictionary represents a row from csv file
    '''
    reader = DictReader(open(args.plco_data_path,"r"))
    rows = [r for r in reader]
    NUM_TRAIN, NUM_VAL = 100000, 25000
    random.seed(0)
    random.shuffle(rows)
    train, val, test = rows[:NUM_TRAIN], rows[NUM_TRAIN:NUM_TRAIN+NUM_VAL], rows[NUM_TRAIN+NUM_VAL:]

    return train, val, test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

import matplotlib.pyplot as plt

def plot_loss_curves(model, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Training Loss')
    if model.val_losses:
        plt.plot(model.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_model_performance_with_nlst(y_true, y_pred_proba, nlst_predictions):
    """
    Plot ROC and Precision-Recall curves with NLST criteria point highlighted
    
    Args:
        y_true: True binary labels
        y_pred_proba: Model's predicted probabilities
        nlst_predictions: Binary predictions using NLST criteria (from nlst_flag column)
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='blue', label=f'Model (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Calculate and plot NLST point on ROC curve
    nlst_fpr = np.mean(nlst_predictions[y_true == 0])  # False positive rate
    nlst_tpr = np.mean(nlst_predictions[y_true == 1])  # True positive rate
    ax1.plot(nlst_fpr, nlst_tpr, 'ro', markersize=10, 
             label=f'NLST Criteria (TPR={nlst_tpr:.3f}, FPR={nlst_fpr:.3f})')
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, color='blue', label=f'Model (AUC = {pr_auc:.3f})')
    
    # Calculate and plot NLST point on PR curve
    nlst_precision = np.mean(y_true[nlst_predictions == 1])  # Precision
    nlst_recall = np.mean(nlst_predictions[y_true == 1])     # Recall (same as TPR)
    ax2.plot(nlst_recall, nlst_precision, 'ro', markersize=10,
             label=f'NLST Criteria (Precision={nlst_precision:.3f}, Recall={nlst_recall:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_subgroup(df, feature):
    df = df.dropna(subset=feature)
    values = df[feature].unique()
    roc_tracker = []
    for v in values: 
        subset = df[df[feature] == v]
        value_fpr, value_tpr, thresholds = roc_curve(subset['test_Y'], subset['pred_test_Y'])
        value_roc_auc = auc(value_fpr, value_tpr)
        roc_tracker.append(value_roc_auc)
    
    fig = plt.figure(figsize=(10, 6))  
    bars = plt.bar(values, roc_tracker)
    for bar in bars:
        yval = np.round(bar.get_height(),2)
        plt.text(bar.get_x() + bar.get_width() / 2, yval*0.8, yval, 
                ha='center', va='bottom', fontsize=12)
    plt.xlabel(feature)
    plt.ylabel('ROC AUC')
    plt.title('Understanding Subgroup Performance')
    plt.ylim(0, 1) 

    return fig

def main(args: argparse.Namespace) -> dict:
    print(args)
    print("Loading data from {}".format(args.plco_data_path))
    train, val, test = load_data(args)

    nlst_data_dict = DictReader(open('/Users/rbhalerao/Desktop/project1_modified/nlst_agg.csv',"r"))
    nlst_data_pd = pd.read_csv('/Users/rbhalerao/Desktop/project1_modified/nlst_agg.csv')
    # TODO: Define someway to define what features your model should use
    # Load feature configuration from features.json
    with open('features.json', 'r') as f:
        feature_config = json.load(f)
    print(feature_config)

    print("Initializing vectorizer and extracting features")
    # TODO: Implement a vectorizer to convert the age features into a feature vector
    plco_vectorizer = Vectorizer(feature_config, feature_map_path='feature_map.json')
    # TODO: Fit the vectorizer on the training data (i.e. compute means for normalization, etc)
    plco_vectorizer.fit(train)

    # TODO: Featurize the training, validation and testing data
    train_X = plco_vectorizer.transform(train)
    val_X = plco_vectorizer.transform(val)
    test_X = plco_vectorizer.transform(test)

    train_Y = np.array([int(r["lung_cancer"]) for r in train])
    val_Y = np.array([int(r["lung_cancer"]) for r in val])
    test_Y = np.array([int(r["lung_cancer"]) for r in test])

    nlst_dataset_transformed = plco_vectorizer.transform(nlst_data_dict, nlst=True)

    print("Training model")
    class_weights = {0: 1.0, 1: 5.0} 
    # TODO: Initialize and train a logistic regression model
    model = LogisticRegression(num_epochs=args.num_epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, regularization_lambda=args.regularization_lambda, class_weights=class_weights,verbose=True)

    model.fit(train_X, train_Y, val_X, val_Y)

    print("Evaluating model")

    pred_train_Y = model.predict_proba(train_X)
    pred_val_Y = model.predict_proba(val_X)
    pred_nlst = model.predict_proba(nlst_dataset_transformed)

    
    nlst_Y = nlst_data_pd['label'].values
    #Comment this line in only for final model 
    pred_test_Y = model.predict_proba(test_X)

    val_fpr, val_tpr, thresholds = roc_curve(val_Y, pred_val_Y)
    test_fpr, test_tpr, thresholds = roc_curve(test_Y, pred_test_Y)
    val_roc_auc = auc(val_fpr, val_tpr)
    test_roc_auc = auc(test_fpr, test_tpr)
    
    nlst_fpr, nlst_tpr, thresholds = roc_curve(nlst_Y, pred_nlst)
    nlst_roc_auc = auc(nlst_fpr, nlst_tpr)

    results = {
        "train_auc": roc_auc_score(train_Y, pred_train_Y),
        "val_auc": roc_auc_score(val_Y, pred_val_Y), 
        "test_auc": roc_auc_score(test_Y, pred_test_Y),
        "nlst_auc": roc_auc_score(nlst_Y, pred_nlst) #commented in only for final model
    }

    print(results)

    # Plot and save roc and loss curves

    json_path = os.path.join('logs/jsons', f"{args.results_path}.json")
    loss_curve_path = os.path.join('logs/final_curves', f"{args.results_path}_loss_curves.png")

    # Plot validation ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(val_fpr, val_tpr, color='blue', label=f'ROC curve (area = {val_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Validation Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Save Validation ROC curve plot
    val_roc_curve_path = os.path.join('logs/final_curves', f"{args.results_path}_val_roc_curve.png")
    plt.savefig(val_roc_curve_path)
    plt.close()
    print("Validation ROC curve saved to: ", val_roc_curve_path)

    #Plotting test ROC curve
    plt.figure(figsize=(10, 6))
    print(test_roc_auc)
    plt.plot(test_fpr, test_tpr, color='blue', label=f'ROC curve (area = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # Save Validation ROC curve plot
    test_roc_curve_path = os.path.join('logs/final_curves', f"{args.results_path}_test_roc_curve.png")
    plt.savefig(test_roc_curve_path)
    plt.close()
    print("Validation ROC curve saved to: ", test_roc_curve_path)

    #print(loss_curve_path)
    plot_loss_curves(model, loss_curve_path)
    print("Loss curves saved to: ", loss_curve_path)
    
    json.dump(results, open(json_path, "w"), indent=True, sort_keys=True)
    print("Saving results to {}".format(args.results_path))
    
    # Get ground truth and NLST flags
    test_nlst_flags = np.array([int(r["nlst_flag"]) if r["nlst_flag"] != '' else 0 for r in test])
    

    # Create and save the plots
    fig = plot_model_performance_with_nlst(test_Y, pred_test_Y, test_nlst_flags)
    plt.savefig('logs/final_curves/screening_performance.png', dpi=300)
    plt.close()
    print("Done")

    df_test_whole = pd.DataFrame(test)
    df_test_whole["test_Y"] = test_Y
    df_test_whole["pred_test_Y"] = pred_test_Y

    sex_subgroup_fig = plot_subgroup(df_test_whole, 'sex')
    plt.savefig('logs/final_curves/sex_auc_performance.png', dpi=300)
    plt.close()

    race_subgroup_fig = plot_subgroup(df_test_whole, 'race7')
    plt.savefig('logs/final_curves/race_auc_performance.png', dpi=300)
    plt.close()

    edu_subgroup_fig = plot_subgroup(df_test_whole, 'educat')
    plt.savefig('logs/final_curves/education_auc_performance.png', dpi=300)
    plt.close()

    cig_subgroup_fig = plot_subgroup(df_test_whole, 'cig_stat')
    plt.savefig('logs/final_curves/cig_auc_performance.png', dpi=300)
    plt.close()

    nlst_subgroup_fig = plot_subgroup(df_test_whole, 'nlst_flag')
    plt.savefig('logs/final_curves/nlst_flag_auc_performance.png', dpi=300)
    plt.close()

    #def model_predict(X):
    #    return model.predict_proba(X)

    #feature_names = plco_vectorizer.subgroups
    #feature_array = [key for key, count in feature_names.items() for _ in range(count)]
    # Create a SHAP explainer
    #explainer = shap.Explainer(model_predict, train_X)

    # Calculate SHAP values for your test data
    #shap_values = explainer(test_X)  # Replace `test_X` with your actual test data

    # Create a summary plot
    #shap.summary_plot(shap_values, test_X, feature_names = feature_array)

    
    #nlst_df = pd.DataFrame([get_feature_metrics(df_test_whole, 'nlst_flag')], index=['NLST'])
    #model_df = pd.DataFrame([get_feature_metrics(df_test_whole, 'pred_test_Y')], index=['Model'])
    #result_df = pd.concat([nlst_df, model_df], axis=0)
    #result_df = result_df.rename_axis(columns='Variables')

    df_test_whole.to_csv(os.path.join('logs/clinical_utility/', f"{args.results_path}"), index=True)
    print('Saved csv with test pred')


    #Plotting test ROC curve
    plt.figure(figsize=(10, 6))
    print(nlst_roc_auc)
    plt.plot(nlst_fpr, nlst_tpr, color='blue', label=f'ROC curve (area = {test_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Save NLST ROC curve plot
    nlst_roc_curve_path = os.path.join('logs/final_curves', f"{args.results_path}_nlst_roc_curve.png")
    plt.savefig(nlst_roc_curve_path)
    plt.close()
    print("Validation ROC curve saved to: ", test_roc_curve_path)

    return results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
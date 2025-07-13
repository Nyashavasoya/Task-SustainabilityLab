import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
from models.cnn_model import build_cnn_model
from models.lstm import build_convlstm_model

DATASET_DIR = "Dataset"

def load_all_data(dataset_dir):
    data = {}
    files = glob.glob(os.path.join(dataset_dir, "*.parquet"))
    
    if not files:
        print("files not found")
        return data
    
    for file in files:
        participant_id = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_parquet(file)
        # filter  'Body event' labels
        df = df[df["label"].str.lower() != "body event"]
        data[participant_id] = df
    
    return data

def inspect_labels(dataset_dir):
    files = glob.glob(os.path.join(dataset_dir, "*.parquet"))
    all_labels = set()
    
    for file_path in files:
        participant_id = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_parquet(file_path)
        df = df[df["label"].str.lower() != "body event"]
        labels = df["label"].unique()
        all_labels.update(labels)

    return sorted(all_labels)


# encoding and scaling
def preprocess_data(df, scaler=None, label_encoder=None, fit_scaler=False, fit_encoder=False):
    feature_columns = ["mean", "std", "min", "max"]
    X = df[feature_columns].values
    y = df["label"].values
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
    
    if fit_encoder:
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, y_encoded, scaler, label_encoder

# inbalance handling
def calculate_class_weights(y_encoded):
    unique_classes = np.unique(y_encoded)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_encoded
    )
    return dict(zip(unique_classes, class_weights))



def evaluate_model(y_true, y_pred, global_classes):
    present_classes = np.intersect1d(np.unique(y_true), np.unique(y_pred))

    acc = accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=present_classes, zero_division=0)
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    present_class_names = [global_classes[i] for i in present_classes]
    report = classification_report(y_true, y_pred, 
                                 labels=present_classes,
                                 target_names=present_class_names, 
                                 zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'present_classes': present_classes,
        'classification_report': report
    }

def run_lopo_cv(data, model_builder, model_name):
    participants = list(data.keys())
    results = defaultdict(list)
    
    all_data = pd.concat(data.values())
    global_le = LabelEncoder()
    global_le.fit(all_data["label"])
    global_classes = global_le.classes_
    num_classes = len(global_classes)
    
    for fold_idx, test_id in enumerate(participants, 1):
        print(f"\nFold {fold_idx}/{len(participants)}: Testing on {test_id}")
        
        train_data = pd.concat([data[p] for p in participants if p != test_id])
        test_data = data[test_id]
        
        print(f"training samples: {len(train_data)}")
        print(f"testing samples: {len(test_data)}")
        
        X_train, y_train, scaler, _ = preprocess_data(
            train_data, fit_scaler=True, label_encoder=global_le, fit_encoder=False
        )
        
        X_test, y_test, _, _ = preprocess_data(
            test_data, scaler=scaler, label_encoder=global_le, fit_encoder=False
        )
        
        # reshape for models
        X_train = X_train.reshape((X_train.shape[0], 4, 1))
        X_test = X_test.reshape((X_test.shape[0], 4, 1))
        
        class_weights = calculate_class_weights(y_train)
        
        model = model_builder(input_shape=(4, 1), num_classes=num_classes)
        
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0
        )
        
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        eval_results = evaluate_model(y_test, y_pred, global_classes)
        
        results['accuracy'].append(eval_results['accuracy'])
        results['macro_f1'].append(eval_results['macro_f1'])
        results['weighted_f1'].append(eval_results['weighted_f1'])\
        
        print("Metric:")
        print(eval_results['accuracy'])
        print(eval_results['macro_f1'])
        print(eval_results['weighted_f1'])
        print(eval_results['present_classes'])
    
    mean_acc = np.mean(results['accuracy'])
    std_acc = np.std(results['accuracy'])
    mean_macro_f1 = np.mean(results['macro_f1'])
    std_macro_f1 = np.std(results['macro_f1'])
    mean_weighted_f1 = np.mean(results['weighted_f1'])
    std_weighted_f1 = np.std(results['weighted_f1'])
    
    
    return {
        'accuracy': results['accuracy'],
        'macro_f1': results['macro_f1'],
        'weighted_f1': results['weighted_f1'],
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_macro_f1': mean_macro_f1,
        'std_macro_f1': std_macro_f1,
        'mean_weighted_f1': mean_weighted_f1,
        'std_weighted_f1': std_weighted_f1
    }



unique_labels = inspect_labels(DATASET_DIR)

data = load_all_data(DATASET_DIR)

if not data:
    print("no data loaded")
    exit(1)

models = [
    (build_cnn_model, "CNN"),
    (build_convlstm_model, "ConvLSTM")
]


all_results = {}

for model_builder, model_name in models:
    results = run_lopo_cv(data, model_builder, model_name)
    all_results[model_name] = results

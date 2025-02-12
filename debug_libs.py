import pandas
import numpy
import csv

# Currently broken
def label_distrib(data, encoder=None):
    # If input is dataframe, extract the label column
    if isinstance(data, pandas.DataFrame):
        if 'Label' not in data.columns:
            raise ValueError("DataFrame must contain a 'Label' column.")
        label_data = data['Label']
    elif isinstance(data, pandas.Series):
        label_data = data
    else:
        raise TypeError("Inumpyut must be a pandas DataFrame or Series.")
    
    label_counts = label_data.value_counts()
    
    # Decode labels if encoder is provided
    if encoder:
        label_names = [encoder.inverse_transform([label])[0] for label in label_counts.index]
    else:
        label_names = label_counts.index.tolist()
    
    max_label_length = 26  # Ensures static table width
    max_count_length = 9   # Ensures static table width
    
    print("\n======== Label Distribution ========")
    for label, count in zip(label_names, label_counts):
        print(f"{str(label).ljust(max_label_length)} {str(f'{count:,}').rjust(max_count_length)}")


def count_invalid(df, message="none"):
    numeric_df = df.select_dtypes(include=[numpy.number])

    if message == "none":
        print("\n=== Invalid rows by type, column ===")
    else:
        print(f"\n[{message}] Invalid row count by affected column:\n")

    inf_count = numpy.isinf(numeric_df).sum()
    inf_nonzero = inf_count[inf_count > 0]
    print("Inf value(s):\n", inf_nonzero if not inf_nonzero.empty else "None")

    nan_count = numpy.isnan(numeric_df).sum()
    nan_nonzero = nan_count[nan_count > 0]
    print("\nNaN value(s):\n", nan_nonzero if not nan_nonzero.empty else "None")
    

def init_log(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", 
            "train_loss", 
            "train_acc", 
            "train_precision", 
            "train_recall", 
            "test_loss", 
            "test_acc", 
            "test_precision", 
            "test_recall"
        ])


def log_metrics(csv_file, epoch, 
                train_loss, train_acc, train_precision, train_recall, 
                test_loss, test_acc, test_precision, test_recall):
    
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss}",
            f"{train_acc}",
            f"{train_precision}",
            f"{train_recall}",
            f"{test_loss}",
            f"{test_acc}",
            f"{test_precision}",
            f"{test_recall}"
        ])

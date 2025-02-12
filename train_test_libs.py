import torch
import pandas
from sklearn.metrics import precision_score, recall_score

def load_set(dataset):
    df_list = []
    for subset in dataset:
        temp_df = pandas.read_csv(subset)
        df_list.append(temp_df)

    df = pandas.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip() # Spacje

    return df


def train_one_epoch(model, dataloader, criterion, optimizer):

    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0
    all_preds = []
    all_labels = []

    for batch_X, batch_y in dataloader:

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == batch_y).sum().item()
        total_train += batch_y.size(0)
        
        all_preds.append(predicted.detach().cpu())
        all_labels.append(batch_y.detach().cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_train / total_train

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall


def test_one_epoch(model, dataloader, criterion):

    model.eval()
    total_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == batch_y).sum().item()
            total_test += batch_y.size(0)

            all_preds.append(predicted.detach().cpu())
            all_labels.append(batch_y.detach().cpu())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_test / total_test

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, precision, recall, all_preds, all_labels


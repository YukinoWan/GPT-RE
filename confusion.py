from pycm import *
import pandas as pd
from testeval import compute_f1
from sklearn.metrics import classification_report
import sys

def confusion(preds, labels):
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=preds)
    cm.print_matrix()
    cm.print_normalized_matrix()

def extract_stat(pred_file, label_file):
    #preds = pd.read_csv(pred_file)
    #labels = pd.read_csv(label_file)
    with open(pred_file, "r") as f:
        line = f.readline()
        #print(line)
        #print(type(line))
        preds = [int(x.strip()) for x in line.strip("[").strip("]").split(",")]
    print(preds[:10])
    with open(label_file, "r") as f:
        line = f.readline()
        labels = [int(x.strip()) for x in line.strip("[").strip("]").split(",")]
    print(labels[:10])

    return preds, labels
    #assert False

def choose_top(tmp_pred, threshold):

    return max(set(tmp_pred), key=tmp_pred.count)

def find_frequency(tmp_pred, target):
    count = 0
    for item in tmp_pred:
        if item == target:
            count += 1

    return 1.0 * count / len(tmp_pred)

def extract_knn_pred(knn_file, labels, preds, threshold):
    label_key = set(labels)
    knn_pred_per_rel = [[] for i in range(len(label_key)+1)]
    knn_label_per_rel = [[] for i in range(len(label_key)+1)]
    knn_pred_frequency = []
    knn_label_frequency = []
    knn_preds = []
    lineid = 0
    with open(knn_file, "r") as f:
        for line in f.read().splitlines():
            tmp_pred = [int(x.strip()) for x in line.strip("[").strip("]").split(",")]
            label_target = labels[lineid]
            pred_target = preds[lineid]

            choice = choose_top(tmp_pred, threshold)
            pred_frequency = find_frequency(tmp_pred, pred_target)
            label_frequency = find_frequency(tmp_pred, label_target)

            knn_pred_frequency.append(pred_frequency)
            knn_label_frequency.append(label_frequency)

            knn_pred_per_rel[pred_target].append(pred_frequency)
            knn_label_per_rel[label_target].append(label_frequency)

            lineid += 1

            knn_preds.append(choice)
            
            #print(tmp_pred)
            #print(choice)
            #assert False
    
    print("The AVG knn frequency of predictions: {}".format(sum(knn_pred_frequency) / len(knn_pred_frequency)))
    print("The AVG knn frequency of gold labels: {}".format(sum(knn_label_frequency) / len(knn_label_frequency)))
    #print(knn_pred_per_rel)
    for i in range(len(label_key)):
        if len(knn_pred_per_rel[i]) == 0:
            continue
        print("The AVG knn frequency of predictions on relation {}: {}".format(i, sum(knn_pred_per_rel[i])/len(knn_pred_per_rel[i])))
        print("The AVG knn frequency of gold labels on relation {}: {}".format(i, sum(knn_label_per_rel[i])/len(knn_label_per_rel[i])))

    return knn_preds

if __name__ == "__main__":
    stat_path = sys.argv[1]
    pred_file = "{}/preds.csv".format(stat_path)
    label_file = "{}/labels.csv".format(stat_path)
    knn_file = "{}/knn.csv".format(stat_path)

    threshold = 0

    preds, labels = extract_stat(pred_file, label_file)
    knn_preds = extract_knn_pred(knn_file, labels, preds, threshold)

    print("Result of preds and labels:")
    pred_f1 = compute_f1(preds, labels)
    print(pred_f1)
    confusion(preds, labels)
    print("Result of knn_preds and labels:")
    knn_f1 = compute_f1(knn_preds, labels)
    print(knn_f1)
    confusion(knn_preds, labels)

    

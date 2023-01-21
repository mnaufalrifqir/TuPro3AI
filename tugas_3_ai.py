
#import
import numpy as np
import pandas as pd
import random
from pprint import pprint

#data
col_names = ['x1', 'x2', 'x3', 'y']
data = pd.read_excel("traintest.xlsx", names=col_names)
test = pd.read_excel("traintest.xlsx", names=col_names, sheet_name=['test'])

test = pd.concat(test, axis=0, ignore_index=True)

data = data.rename(columns={"y": "label"})
test = test.rename(columns={"y": "label"})

#split trainset(seen data)
def train_val_split(data, val_size):
    
    if isinstance(val_size, float):
        val_size = round(val_size * len(data))

    indices = data.index.tolist()
    test_indices = random.sample(population=indices, k=val_size)

    val_data = data.loc[test_indices]
    train_data = data.drop(test_indices)
    
    return train_data, val_data

random.seed(0)
train_set, test_set = train_val_split(data, val_size=20)

#cek if homogenous
def ismono(data):
  label_column = data[:, -1]
  uc = np.unique(label_column)

  if len(uc) == 1:
    return True
  else:
    return False

#data classification
def classify(data):
    
    label_column = data[:, -1]
    u_class, count_uclass = np.unique(label_column, return_counts=True)


    index = count_uclass.argmax()
    classification = u_class[index]
    
    return classification

#split
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):       
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_bawah = data[split_column_values <= split_value]
        data_atas = data[split_column_values >  split_value]
    else:
        data_bawah = data[split_column_values == split_value]
        data_atas = data[split_column_values != split_value]
    
    return data_bawah, data_atas

#entropy & info gain
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


def cal_info_gain(data_bawah, data_atas):
    
    n = len(data_bawah) + len(data_atas)
    p_data_bawah = len(data_bawah) / n
    p_data_atas = len(data_atas) / n

    info_gain =  (p_data_bawah * calculate_entropy(data_bawah) 
                      + p_data_atas * calculate_entropy(data_atas))
    
    return info_gain

#best split
def best_split(data, potential_splits):
    
    info_gain = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_bawah, data_atas = split_data(data, split_column=column_index, split_value=value)
            curr_info_gain = cal_info_gain(data_bawah, data_atas)

            if curr_info_gain <= info_gain:
                info_gain = curr_info_gain
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

def determine_type_of_feature(data):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in data.columns:
        if feature != "label":
            unique_values = data[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

#main procedure
def decision_tree_algorithm(data, counter=0, min_samples=2, max_depth=3):
    
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = data.columns
        FEATURE_TYPES = determine_type_of_feature(data)
        data = data.values
    else:
        data = data           
    
    
    # cek kalo homogenous atau data kurand dari minimum sample atau udh dalam kedalaman yang di inginkan
    if (ismono(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify(data)
        
        return classification

    
    else:    
        counter += 1

        potential_splits = get_potential_splits(data)
        split_column, split_value = best_split(data, potential_splits)
        data_bawah, data_atas = split_data(data, split_column, split_value)
        
        # check jika data kosong
        if len(data_bawah) == 0 or len(data_atas) == 0:
            classification = classify(data)
            return classification
        

        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        
        sub_tree = {question: []}
        
        positive = decision_tree_algorithm(data_bawah, counter, min_samples, max_depth)
        negative = decision_tree_algorithm(data_atas, counter, min_samples, max_depth)
        

        if positive == negative:
            sub_tree = positive
        else:
            sub_tree[question].append(positive)
            sub_tree[question].append(negative)
        
        return sub_tree

#predict/classify
def predict(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    if comparison_operator == "<=":  # feature is continuous
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer
    
    else:
        residual_tree = answer
        return predict(example, residual_tree)

#itung akurasi
def calculate_accuracy(data, tree):

    data["classification"] = data.apply(predict, axis=1, args=(tree,))
    data["classification_correct"] = data["classification"] == data["label"]
    
    accuracy = data["classification_correct"].mean()
    
    return accuracy

#print tree
depth = int(input("masukkan depth maksimum yang ingin di telusuri: "))
tree = decision_tree_algorithm(train_set, max_depth=depth)
pprint(tree)

test_result = []

for i in range(len(test_set)):
  test_result.append(predict(test_set.iloc[i], tree))

print("hasil test_set: ", test_result)

#cek akurasi
accuracy = calculate_accuracy(test_set, tree)
print("akurasi test_set: ", accuracy)

train_acc = train_set

train_accuracy = calculate_accuracy(train_acc, tree)
print("akurasi train_set: ", train_accuracy)


#predict unseen data
result = []
for i in range(len(test)):
  result.append(predict(test.iloc[i], tree))

print("hasil unseen data: ", result)
test['label'] = result


test.to_excel('hasil.xlsx')

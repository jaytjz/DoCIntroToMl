import numpy as np
from node import Node
from numpy.random import default_rng
        
def find_entropy(class_column):
    classes, counts = np.unique(class_column,return_counts=True)
    entropy = (-counts/np.sum(counts)*np.log2(counts/np.sum(counts))).sum()
    return entropy

def find_split(training_dataset):
    classes = training_dataset.transpose()[-1]
    total_entropy = find_entropy(classes)
    max_info_gain = (0,0,0)
    for column_index,column in enumerate(training_dataset.transpose()[:-1]):
        sorted_indicies = np.argsort(column)
        sorted_column = column[sorted_indicies]
        sorted_classes = classes[sorted_indicies]
        for i in range(1, len(sorted_indicies)):
            if(sorted_column[i] != sorted_column[i-1]):

                split_value = (sorted_column[i] + sorted_column[i - 1]) / 2

                l_weighted_entropy = (i/len(classes))*find_entropy(sorted_classes[:i])
                r_weighted_entropy = ((len(classes)-i)/len(classes))*find_entropy(sorted_classes[i:])
                information_gain = total_entropy - (l_weighted_entropy+r_weighted_entropy)

                if information_gain > max_info_gain[0]:
                    max_info_gain = (information_gain,column_index,split_value)
    return max_info_gain

def decision_tree_learning(training_dataset,depth=0):
    classes = training_dataset.transpose()[-1]
    if (np.all(classes == classes[0])):
        return (Node(-1,classes[0],None,None, True),depth)
    information_gain, column_index, split_value = find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(training_dataset[training_dataset[:,column_index] < split_value] ,depth+1)
    r_branch, r_depth = decision_tree_learning(training_dataset[training_dataset[:,column_index] >= split_value],depth+1)
    return (Node(column_index,split_value,l_branch,r_branch,False), max(l_depth,r_depth))
    
def train(data, test_proportion, seed):
    shuffled_indices = default_rng(seed).permutation(len(data))
    n_test = round(len(data) * test_proportion)
    n_train = len(data) - n_test
    training_data = data[shuffled_indices[:n_train]]
    testing_data = data[shuffled_indices[n_train:]]

    root, depth = decision_tree_learning(training_data)
    print(f"Depth data {depth}")
    root.prune_tree()
    return (root, testing_data)

def test(root,test_data):
    correct_predictions = 0
    for instance in test_data:
        prediction = root.traverse_tree(instance)
        if prediction == instance[-1]:
            correct_predictions += 1
    return correct_predictions/len(test_data)

if __name__ == "__main__":
    clean = np.loadtxt("clean_dataset.txt")
    noisy = np.loadtxt("noisy_dataset.txt")
    clean_root, clean_testing_data = train(clean,0.2,4)
    print(f"Clean decision tree accuracy on clean test data = {100*test(clean_root,clean_testing_data)}%")
    print(f"Clean decision tree accuracy on noisy test data = {100*test(clean_root,noisy)}%")

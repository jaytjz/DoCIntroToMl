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

def decision_tree_learning(training_dataset, depth=0):
    classes = training_dataset.transpose()[-1]
    if (np.all(classes == classes[0])):
        return (Node(-1,classes[0],None,None, True),depth)
    information_gain, column_index, split_value = find_split(training_dataset)
    l_branch, l_depth = decision_tree_learning(training_dataset[training_dataset[:,column_index] < split_value] ,depth+1)
    r_branch, r_depth = decision_tree_learning(training_dataset[training_dataset[:,column_index] >= split_value],depth+1)
    return (Node(column_index,split_value,l_branch,r_branch,False), max(l_depth,r_depth))

def test(root, test_data):
    correct_predictions = 0
    for instance in test_data:
        prediction = root.traverse_tree(instance)
        if prediction == instance[-1]:
            correct_predictions += 1
    return correct_predictions/len(test_data)
   
def train(data, test_proportion, prune_proportion, seed):
def train(data, test_proportion, prune_proportion, seed):
    shuffled_indices = default_rng(seed).permutation(len(data))
    n_test = round(len(data) * test_proportion)
    n_prune = round(len(data)*prune_proportion)
    n_train = len(data) - (n_test+n_prune)
    n_prune = round(len(data)*prune_proportion)
    n_train = len(data) - (n_test+n_prune)
    training_data = data[shuffled_indices[:n_train]]
    pruning_data = data[shuffled_indices[n_train:n_train+n_prune]]
    testing_data = data[shuffled_indices[n_train+n_prune:]]
    pruning_data = data[shuffled_indices[n_train:n_train+n_prune]]
    testing_data = data[shuffled_indices[n_train+n_prune:]]

    root, depth = decision_tree_learning(training_data)
    print(f"Depth data {depth}")
    return (root, testing_data, pruning_data, depth)
    return (root, testing_data, pruning_data, depth)

if __name__ == "__main__":
    clean = np.loadtxt("clean_dataset.txt")
    noisy = np.loadtxt("noisy_dataset.txt")
    min_depth = 1000000000
    # This is just for me because I wanted to test the visualisation at different depths no need to iterate through like this to make a decision tree
    # for i in range(1):
    clean_root, clean_testing_data, clean_pruning_data, clean_depth = train(clean,0.1,0.3,10)
    noisy_root, noisy_testing_data, noisy_pruning_data, noisy_depth = train(noisy,0.1,0.3,10)
        # if clean_depth < min_depth:
        #     min_root = clean_root
        #     min_depth = cleandepth
        #     min_testing_data = clean_testing_data
    
    # clean_root.draw_tree()
    print(f"Clean decision tree accuracy on clean test data before aggressive pruning = {100*test(clean_root,clean_testing_data)}%")
    print(f"Noisy decision tree accuracy on noisy test data before aggressive pruning = {100*test(noisy_root,noisy_testing_data)}%")
    clean_root.prune_until_converged(clean_pruning_data,clean_root,test)
    noisy_root.prune_until_converged(noisy_pruning_data,noisy_root,test)
    print(f"Clean decision tree accuracy on clean test data after aggressive pruning = {100*test(clean_root,clean_testing_data)}%")
    print(f"Noisy decision tree accuracy on noisy test data after aggressive pruning = {100*test(noisy_root,noisy_testing_data)}%")
    print(f"Clean decision tree accuracy on clean test data before aggressive pruning = {100*test(clean_root,clean_testing_data)}%")
    print(f"Noisy decision tree accuracy on noisy test data before aggressive pruning = {100*test(noisy_root,noisy_testing_data)}%")
    clean_root.prune_until_converged(clean_pruning_data,clean_root,test)
    noisy_root.prune_until_converged(noisy_pruning_data,noisy_root,test)
    print(f"Clean decision tree accuracy on clean test data after aggressive pruning = {100*test(clean_root,clean_testing_data)}%")
    print(f"Noisy decision tree accuracy on noisy test data after aggressive pruning = {100*test(noisy_root,noisy_testing_data)}%")
    clean_root.draw_tree()
    noisy_root.draw_tree()
    noisy_root.draw_tree()

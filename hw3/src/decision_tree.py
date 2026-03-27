"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np
feature_name = {}

def class_counts(rows): 
    counts = {}  
    for row in rows:
        label = row[-1] 
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Question: 
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column] 
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row) 
    return true_rows, false_rows

def Gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def info_gain_gini(left, right, current_uncertainty): 
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * Gini(left) - (1 - p) * Gini(right)

def find_best_split(rows, criterion):
    best_gain = 0  
    best_question = None  
    current_uncertainty = Gini(rows) 
    n_features = len(rows[0]) - 1  

    for col in range(n_features): 
        values = set([row[col] for row in rows]) 
        for val in values:  
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question) 
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain_gini(true_rows, false_rows, current_uncertainty) 

            if gain >= best_gain: 
                best_gain, best_question = gain, question
    return best_gain, best_question

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None): 
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
    def fit(self, x_data, y_data):

        
        if y_data.ndim == 1:
            Y_data = np.reshape(y_data, (-1, 1))
        else:
            Y_data = y_data.copy()

       
        data = np.hstack((x_data, Y_data))
        tree = build_tree(data, self.criterion, self.max_depth) 
        self.root = tree
    def predict(self, xtest): 
        y_pred = []
        for row in xtest:
            dic={}
            dic = print_leaf(classify(row, self.root))  
            if 1.0 not in dic:
                dic[1.0] = 0
            elif 0.0 not in dic:
                dic[0.0] = 0
            
            if dic[1.0] > dic[0.0]:
                y_pred.append(1)
            else:
                y_pred.append(0)
        return y_pred
    def compute_feature_importance(self, feature_names):
        feature_importance = {}
    
        def traverse_tree(node, importance):
            nonlocal feature_importance
            
            if isinstance(node, Leaf):
                return
            
            feature = feature_names[node.question.column]
            importance[feature] = importance.get(feature, 0) + 1
            
            traverse_tree(node.true_branch, importance)
            traverse_tree(node.false_branch, importance)
        
        traverse_tree(self.root, feature_importance)

        
        return feature_importance

        
class Leaf: 
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows, criterion, depth=None):
    global feature_name
    gain, question = find_best_split(rows, criterion) 
    if gain == 0:
        return Leaf(rows)
        
    num = question.column 
    cou = 0
    for i in feature_name:
        if cou == num:
            feature_name[i] += 1
        cou += 1

    true_rows, false_rows = partition(rows, question)
    depth-=1
    if depth > 0:
        true_branch = build_tree(true_rows, criterion, depth)
        false_branch = build_tree(false_rows, criterion, depth)
        return Node(question, true_branch, false_branch)
    else: 
        true_branch = Leaf(true_rows)
        false_branch = Leaf(false_rows)
        return Node(question, true_branch, false_branch)
    
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row): 
        return classify(row, node.true_branch)
    else: 
        return classify(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = (counts[lbl] / total * 100)
    
    return probs

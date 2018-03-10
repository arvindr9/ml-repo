training_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

categories = ['color', 'diameter', 'label']

def frequencies(rows):
    freq = {}
    for row in rows:
        label = row[-1]
        if label not in freq:
            freq[label] = 1
        else:
            freq[label] += 1
    return freq


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
    def match(self, row):
        if isinstance(self.value, int) or isinstance(self.value, float):
            return self.value >= row[self.column]
        else:
            return self.value == row[self.column]

def partition(rows, question):
    true_rows = []
    false_rows = []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def gini(rows):
    ratio = float(1)
    num_rows = len(rows)
    freq = frequencies(rows)
    for _, v in freq.items():
        ratio -= (v / num_rows)**2
    return ratio

def info_gain(true_rows, false_rows, uncertainty):
    p = len(true_rows) / (len(false_rows) + len(true_rows))
    return uncertainty - p * gini(true_rows) - (1 - p) * gini(false_rows)

def find_best_question(rows):
    num_col = len(rows[0]) - 1
    max_gain = 0
    best_question = None
    uncertainty = gini(rows)
    for col in range(num_col):
        for row in rows:    
            question = Question(col, row[col])
            true_rows, false_rows = partition(rows, question)    
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, uncertainty)    
            if gain > max_gain:
                max_gain = gain
                best_question = question
    return max_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = frequencies(rows)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_question(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")



def main():
    tree = build_tree(training_data)
    print_tree(tree)


if __name__ == '__main__':
    main()




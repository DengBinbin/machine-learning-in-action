# coding: utf-8
import math


class Node:
    def __init__(self, divided_by=None, condition=None, sons=[], label=None):
        self.divided_by = divided_by  # node is divided by this, None if it's leaf
        self.condition = condition  # brunch condition, a float if divided_by type is float, else a list map to sons
        self.sons = sons  # pointer of this node's sons
        self.label = label  # None if it's not leaf, else classification result of this leaf node


def cal_ent(count, total_cnt):
    if total_cnt == 0:
        return 0
    prob = [float(count[k]) / total_cnt for k in count]
    ent = -sum([math.log(i, 2) * i for i in prob])  # log(n[, base] )
    return ent


def cal_info_gain(entd, divided_res, total_cnt, Y):
    if total_cnt == 0:
        return 0
        # print entd
    # for i in divided_res:
    #   print i
    # print total_cnt
    return entd - sum(float(len(data)) / total_cnt * cal_ent(cal_count(data, Y), len(data)) for data in divided_res)


def cal_count(data_rem, Y):
    count = {}
    for i in data_rem:
        kind = Y[i]
        count[kind] = count.get(kind, 0) + 1
    return count


def build_tree(attr_rem, data_rem, X, Y, attr_dict, trees, fa_label):
    # if attr_rem is None:
    #   print None
    # else:
    #   for i in attr_rem:
    #       print i,
    #   print
    # print data_rem
    # print fa_label
    divided_by = None
    sons = []
    label = None
    condition = None

    count = cal_count(data_rem, Y)  # number of every kind
    total_cnt = len(data_rem)  # total number of remain data
    max_kind = max(count, key=lambda x: count[x])  # kind which has the largest number
    ent = cal_ent(count, total_cnt)

    if len(data_rem) == 0:
        label = fa_label
    elif attr_rem is None or len(
            count) == 1:  # remain attribute is empty or all data is one same kind, then this is a leaf node
        label = max_kind
    else:
        best_divided_res = []
        best_condition = []
        best_divided_by = None
        best_info_gain = -100

        for attr in attr_rem:
            is_float = type(X[data_rem[0]][attr]) == float
            if is_float:
                points = sorted(attr_dict[attr])
                divided_points = [(points[i] + points[i + 1]) / 2 for i in range(len(points) - 1)]
                for divided_point in divided_points:
                    divided_res = []
                    divided_res.append([i for i in data_rem if X[i][attr] <= divided_point])
                    divided_res.append([i for i in data_rem if X[i][attr] > divided_point])
                    info_gain = cal_info_gain(ent, divided_res, total_cnt, Y)
                    if info_gain > best_info_gain:
                        best_info_gain = info_gain
                        best_divided_res = divided_res
                        best_divided_by = attr
                        best_condition = divided_point
            else:
                divided_res = []
                for name in attr_dict[attr]:
                    divided_res.append([j for j in data_rem if X[j][attr] == name])
                info_gain = cal_info_gain(ent, divided_res, total_cnt, Y)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_divided_res = divided_res
                    best_divided_by = attr
                    best_condition = attr_dict[attr]

        divided_by = best_divided_by
        is_float = type(X[data_rem[0]][divided_by]) == float
        if not is_float:
            attr_rem = [i for i in attr_rem if
                        i != divided_by]  # can't use remove! behaviour not expected when dealing with utf-8

        sons = [build_tree(attr_rem, i, X, Y, attr_dict, trees, max_kind) for i in best_divided_res]
        condition = best_condition

    trees.append(Node(divided_by, condition, sons, label))
    return len(trees) - 1


def fit(X, Y):
    attr_dict = {}
    for i in range(len(X[0])):
        attr_dict[X[0][i]] = list(set([X[j][i] for j in range(1, len(X))]))  # details of all attributes
    attr_rem = X[0]  # remain attribute on this node
    data_rem = range(1, len(X))  # index of remain data on this node
    X = [dict(zip(X[0], row)) for row in X]  # convert to more convinient data structure
    # for i in X:
    #   for j in i.items():
    #       print j[0], j[1]
    #   print
    trees = []  # records of the built tree
    root = build_tree(attr_rem, data_rem, X, Y, attr_dict, trees, None)  # build the tree recursively
    display(trees, root)


def display(trees, root):
    que = [root]
    while (que):
        pos = que[0]
        del que[0]
        divided_by = trees[pos].divided_by
        condition = trees[pos].condition
        sons = trees[pos].sons
        label = trees[pos].label
        print 'id:'
        print pos
        if divided_by is None:
            print 'label'
            print label
        else:
            print 'divided_by'
            print divided_by
            print 'condition:'
            if type(condition) == list:
                for i in condition:
                    print i,
                print
            else:
                print condition
            print 'sons:'
            for i in sons:
                print i,
            print
            for i in sons:
                que.append(i)
        print


input_path = "data/西瓜数据集3.csv"
file = open(input_path.decode('utf-8'))
filedata = [line.strip('\n').split(',') for line in file]
filedata = [[float(i) if '.' in i.decode('utf-8') else i for i in row] for row in
            filedata]  # change decimal from string to float

X = [row[1:-1] for row in filedata]  # attributes
Y = [row[-1] for row in filedata]  # class label
fit(X, Y)
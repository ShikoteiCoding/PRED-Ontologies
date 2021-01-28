import itertools


def check_gap(mat, mingap=0, maxgap=0):
    for i, mat_i in enumerate(mat):
        if i == 0:
            pre = mat_i
            continue
        if mat_i <= pre + mingap or mat_i > pre + 1 + maxgap:
            return False
        pre = mat_i
    return True


def sequenceMatching(edgeMatching, mingap=0, maxgap=0):
    all_combinations = list(itertools.product(*edgeMatching))
    all_matchings = []
    for mat in all_combinations:
        if check_gap(mat, mingap, maxgap):
            all_matchings.append(list(mat))
    return all_matchings


def relevant_pattern(pattern, patterns, min_gap=0, max_gap=0):
    for b_pat in patterns:
        if is_subsequence(b_pat, pattern, min_gap, max_gap):
            return True
    return False


def is_subsequence(pat1, pat2, mingap=0, maxgap=0):
    pat1_itemsets = [edge.replace("[", "").replace("(","").split(", ") for edge in pat1.split(")")][:-1]
    pat2_itemsets = [edge.replace("[", "").replace("(", "").split(", ") for edge in pat2.split(")")][:-1]
    matching = []
    for itemset1 in pat1_itemsets:
        j = 0
        edge_matching = []
        for itemset2 in pat2_itemsets:
            if all(x in itemset2 for x in itemset1):
                edge_matching.append(j)
            j += 1
        if len(edge_matching) == 0:
            return False
        matching.append(edge_matching)
    matchings = sequenceMatching(matching, mingap, maxgap)
    if len(matchings) == 0:
        return False
    return True

def read_pattern_by_dataset_dic(pat, dataset_dict):
    (path, sup) = pat.split(':')
    p = "["
    elements = str(path).strip().split(" ")
    for ele in elements:
        el = "("
        items = ele.strip().replace('{', "").replace('}', "").split(",")
        items2 = []
        for item in items:
            el += dataset_dict[int(item.strip())] + ", "
            items2.append(dataset_dict[int(item.strip())])
        el = el[:-2]
        el += ")"
        p += el
    p += "]"
    return p, int(sup.strip())



def read_dataset_dictionary(file):
    di = dict()
    with open(file, "r") as f:
        for line in f:
            s, nb = line.strip().split("\t")
            di[int(nb)] = s
    return di

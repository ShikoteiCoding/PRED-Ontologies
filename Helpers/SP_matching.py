import gc
from typing import Iterator, Tuple, List

from Helpers import HyperHypoCouple as Hh
from Helpers import core_functions as cf
from Helpers import spm_core_functions as spm_cf
import itertools

from Helpers.HyperHypoCouple import HHCouple
from Helpers.ParsedSentence import ParsedSentence


def edges_matching(patternEdge, pathEdge) -> bool:
    hyp_list = ["hypo", "hyper", "hyper_dep", "hypo_dep"]
    for patternItem in patternEdge:
        if patternItem in hyp_list:
            continue
        if patternItem not in pathEdge:
            return False
    return True


def get_lemma_from_edge(edge):
    for item in edge:
        if str(item).__contains__("_lemma"):
            return str(item).replace("_lemma", "")
    return ""


def get_label_from_edge(edge):
    for item in edge:
        if str(item).__contains__("_label"):
            return str(item).replace("_label", "").replace("_", " ")
    return ""


def get_dep_from_edge(edge):
    for item in edge:
        if str(item).__contains__("_dep"):
            return str(item).replace("_dep", "")


def spm_matching(pattern: str, parsed_sentence: str, mingap=0, maxgap=3) -> Tuple[bool, List[HHCouple], str]:
    hhCouples = []
    pathEdges = [edge.replace("[", "").replace("(", "").replace(")]", "").split(", ") for edge in
                 parsed_sentence.split("), ")]
    patternEdges = [edge.replace("[", "").replace("(", "").split(", ") for edge in pattern.split(")")][:-1]
    edgeMatching = []
    hypoIndex = -1
    hyperDep = []
    hypoDep = []
    hyperIndex = -1
    i = 0
    min_j = 0
    max_j = len(pathEdges)
    for patternEdge in patternEdges:
        if "hypo" in patternEdge:
            hypoIndex = i
        elif "hyper" in patternEdge:
            hyperIndex = i
        if "hypo_dep" in patternEdge:
            hypoDep.append(i)
        elif "hyper_dep" in patternEdge:
            hyperDep.append(i)
        j = min_j
        li = []
        for pathEdge in itertools.islice(pathEdges, min_j, min(len(pathEdges), max_j + 1)):
            if edges_matching(patternEdge, pathEdge):
                li.append(j)
            j += 1
        if len(li) == 0:
            return False, [], ""
        else:
            min_j = min(li) + 1 + mingap
            max_j = max(li) + 1 + maxgap
        edgeMatching.append(li)
        i += 1
    if hypoIndex == -1 or hyperIndex == -1:
        return False, [], ""
    matchings = spm_cf.sequenceMatching(edgeMatching, mingap, maxgap)
    if len(matchings) == 0:
        return False, [], ""
    for matching in matchings:
        hyperEdge = pathEdges[matching[hyperIndex]]
        hyper = get_lemma_from_edge(hyperEdge)
        hyperNP = cf.remove_first_occurrences_stopwords(get_label_from_edge(hyperEdge))
        flag = False
        for i in hyperDep:
            depHyperEdge = pathEdges[matching[i]]
            dephyper = get_dep_from_edge(depHyperEdge)
            if hyper.strip() == dephyper.strip():
                flag = True
            else:
                flag = False
                break
        if not flag and len(hyperDep) > 0:
            continue
        hypoEdge = pathEdges[matching[hypoIndex]]
        hypo = get_lemma_from_edge(hypoEdge)
        hypoNP = cf.remove_first_occurrences_stopwords(get_label_from_edge(hypoEdge))
        flag2 = False
        for j in hypoDep:
            depHypoEdge = pathEdges[matching[j]]
            dephypo = get_dep_from_edge(depHypoEdge)
            if hypo.strip() == dephypo.strip():
                flag2 = True
            else:
                flag2 = False
                break
        if not flag2 and len(hypoDep) > 0:
            continue
        hhc = Hh.HHCouple(hypoNP, hyperNP)
        if hhc in hhCouples or hhc.hyponym == hhc.hypernym or hhc.hyponym == "" or hhc.hypernym == "":
            continue
        hhCouples.append(hhc)
        cohypos = co_hyponym_extraction(pathEdges, hypo, matching[hypoIndex])
        for chypo in cohypos:
            if chypo == hyperNP:
                continue
            hhc = Hh.HHCouple(chypo, hyperNP)
            if hhc in hhCouples or hhc.hyponym == hhc.hypernym or hhc.hyponym == "" or hhc.hypernym == "":
                continue
            hhCouples.append(hhc)

    if len(hhCouples) == 0:
        return False, [], ""
    return True, hhCouples, pattern


def co_hyponym_extraction(pathEdges, hypo_lemma, hypo_index) -> List:
    coHypos = []
    for i, pEdge in enumerate(pathEdges):
        if i > hypo_index:
            if (str(hypo_lemma) + "_dep") in pEdge and (
                    "conj:and<--" in pEdge or "conj:or<--" in pEdge or "appos<--" in pEdge):
                hypo_NP = pEdge[0].replace("_label", "").replace("_", " ")
                hypo_NP = cf.remove_first_occurrences_stopwords(hypo_NP)
                coHypos.append(hypo_NP)
    return coHypos

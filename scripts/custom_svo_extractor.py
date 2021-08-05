#!/usr/bin/env python3

"""
Methodology inspired from:
https://github.com/peter3125/enhanced-subject-verb-object-extraction.git
"""
import en_core_web_sm
from collections.abc import Iterable

# use spacy small model
nlp = en_core_web_sm.load()

# dependency markers for subjects
SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}
# dependency markers for objects
OBJECTS = {"dobj", "dative", "attr", "oprd"}
# POS tags that will break adjoining items
BREAKER_POS = {"CCONJ", "VERB"}
# words that are negations
NEGATIONS = {"no", "not", "n't", "never", "none"}


def contains_conj(dep_set):
    """
    Check if dependency set contains any coordinating conjunctions
    """
    return "and" in dep_set or "or" in dep_set or "nor" in dep_set or \
           "but" in dep_set or "yet" in dep_set or "so" in dep_set or "for" in dep_set


def _get_subs_from_conjunctions(subs):
    """
    Get subjects joined by conjunctions
    """
    more_subs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        right_deps = {tok.lower_ for tok in rights}
        if contains_conj(right_deps):
            more_subs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(more_subs) > 0:
                more_subs.extend(_get_subs_from_conjunctions(more_subs))
    return more_subs


def _get_objs_from_conjunctions(objs):
    """
    Get objects joined by conjunctions
    """
    more_objs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if contains_conj(rightDeps):
            more_objs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(more_objs) > 0:
                more_objs.extend(_get_objs_from_conjunctions(more_objs))
    return more_objs


def _find_subs(tok):
    """
    Find subject dependencies
    """
    head = tok.head
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verb_negated = _is_negated(head)
            subs.extend(_get_subs_from_conjunctions(subs))
            return subs, verb_negated
        elif head.head != head:
            return _find_subs(head)
    elif head.pos_ == "NOUN":
        return [head], _is_negated(tok)
    return [], False


def _is_negated(tok):
    """
    Check if the token set is left or right negated
    """
    parts = list(tok.lefts) + list(tok.rights)
    for dep in parts:
        if dep.lower_ in NEGATIONS:
            return True
    return False


def _find_svs(tokens):
    """
    Get all the verbs on tokens with negation marker
    """
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append((sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


def _get_objs_from_prepositions(deps, is_pas):
    """
    Get grammatical objects for a given set of dependencies (including passive sentences)
    """
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or (is_pas and dep.dep_ == "agent")):
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS or
                         (tok.pos_ == "PRON" and tok.lower_ == "me") or
                         (is_pas and tok.dep_ == 'pobj')])
    return objs


def _get_objs_from_attrs(deps, is_pas):
    """
    Get objects from the dependencies using the attribute dependency
    """
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(_get_objs_from_prepositions(rights, is_pas))
                    if len(objs) > 0:
                        return v, objs
    return None, None


def _get_obj_from_xcomp(deps, is_pas):
    """
    xcomp; open complement - verb has no subject
    """
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(_get_objs_from_prepositions(rights, is_pas))
            if len(objs) > 0:
                return v, objs
    return None, None


def _get_all_subs(v):
    """
    Get all functional subjects adjacent to the verb passed in
    """
    verb_negated = _is_negated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(_get_subs_from_conjunctions(subs))
    else:
        foundSubs, verb_negated = _find_subs(v)
        subs.extend(foundSubs)
    return subs, verb_negated


def _find_verbs(tokens):
    """
    Find the main verb - or any aux verb if we can't find it
    """
    verbs = [tok for tok in tokens if _is_non_aux_verb(tok)]
    if len(verbs) == 0:
        verbs = [tok for tok in tokens if _is_verb(tok)]
    return verbs


# is the token a verb?  (excluding auxiliary verbs)
def _is_non_aux_verb(tok):
    """
    Is the token a verb?  (excluding auxiliary verbs)
    """
    return tok.pos_ == "VERB" and (tok.dep_ != "aux" and tok.dep_ != "auxpass")


# is the token a verb?  (excluding auxiliary verbs)
def _is_verb(tok):
    return tok.pos_ == "VERB" or tok.pos_ == "AUX"


def _right_of_verb_is_conj_verb(v):
    """
    Return the verb to the right of this verb in a CCONJ relationship if applicable
    Returns a tuple, first part True|False and second part the modified verb if True
    """
    # rights is a generator
    rights = list(v.rights)

    # VERB CCONJ VERB (e.g. he beat and hurt me)
    if len(rights) > 1 and rights[0].pos_ == 'CCONJ':
        for tok in rights[1:]:
            if _is_non_aux_verb(tok):
                return True, tok

    return False, v


def _get_all_objs(v, is_pas):
    """
    Get all objects for an active/passive sentence
    """
    # rights is a generator
    rights = list(v.rights)

    objs = [tok for tok in rights if tok.dep_ in OBJECTS or (is_pas and tok.dep_ == 'pobj')]
    objs.extend(_get_objs_from_prepositions(rights, is_pas))

    # potentialNewVerb, potentialNewObjs = _get_objs_from_attrs(rights)
    # if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
    #    objs.extend(potentialNewObjs)
    #    v = potentialNewVerb

    potential_new_verb, potential_new_objs = _get_obj_from_xcomp(rights, is_pas)
    if potential_new_verb is not None and potential_new_objs is not None and len(potential_new_objs) > 0:
        objs.extend(potential_new_objs)
        v = potential_new_verb
    if len(objs) > 0:
        objs.extend(_get_objs_from_conjunctions(objs))
    return v, objs


def _is_passive(tokens):
    """
    Return true if the sentence is passive - at he moment a sentence is assumed passive if it has an auxpass verb
    """
    for tok in tokens:
        if tok.dep_ == "auxpass":
            return True
    return False


def _get_that_resolution(toks):
    """
    Resolve a 'that' where/if appropriate
    """
    for tok in toks:
        if 'that' in [t.orth_ for t in tok.lefts]:
            return tok.head
    return None


def _get_lemma(word: str):
    """
    Simple stemmer using lemmas
    """
    tokens = nlp(word)
    if len(tokens) == 1:
        return tokens[0].lemma_
    return word


def print_deps(toks):
    """
    Print information for displaying all kinds of things of the parse tree
    """
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [t.orth_ for t in tok.lefts],
              [t.orth_ for t in tok.rights])


def expand(item, tokens, visited):
    """
    Expand an obj / subj np using its chunk
    """
    if item.lower_ == 'that':
        temp_item = _get_that_resolution(tokens)
        if temp_item is not None:
            item = temp_item

    parts = []

    if hasattr(item, 'lefts'):
        for part in item.lefts:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    parts.append(item)

    if hasattr(item, 'rights'):
        for part in item.rights:
            if part.pos_ in BREAKER_POS:
                break
            if not part.lower_ in NEGATIONS:
                parts.append(part)

    if hasattr(parts[-1], 'rights'):
        for item2 in parts[-1].rights:
            if item2.pos_ == "DET" or item2.pos_ == "NOUN":
                if item2.i not in visited:
                    visited.add(item2.i)
                    parts.extend(expand(item2, tokens, visited))
            break

    return parts


def to_str(tokens):
    """
    Convert a list of tokens to a string
    """
    if isinstance(tokens, Iterable):
        return ' '.join([item.text for item in tokens])
    else:
        return ''


def find_svo(text):
    """
    Find verbs and their subjects / objects to create SVOs,
    detect passive/active sentences
    """
    tokens = nlp(text)
    svos = []
    is_pas = _is_passive(tokens)
    verbs = _find_verbs(tokens)
    visited = set()  # recursion detection
    for v in verbs:
        subs, verbNegated = _get_all_subs(v)
        # hopefully there are subs, if not, don't examine this verb any longer
        if len(subs) > 0:
            isConjVerb, conjV = _right_of_verb_is_conj_verb(v)
            if isConjVerb:
                v2, objs = _get_all_objs(conjV, is_pas)
                for sub in subs:
                    for obj in objs:
                        objNegated = _is_negated(obj)
                        if is_pas:  # reverse object / subject for passive
                            to_add = [to_str(expand(obj, tokens, visited)),
                                      "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                      to_str(expand(sub, tokens, visited))]
                            if len(to_add) > 2:
                                svos.append(to_add)
                            to_add = [to_str(expand(obj, tokens, visited)),
                                      "!" + v2.lemma_ if verbNegated or objNegated else v2.lemma_,
                                      to_str(expand(sub, tokens, visited))]
                            if len(to_add) > 2:
                                svos.append(to_add)
                        else:
                            to_add = [to_str(expand(sub, tokens, visited)),
                                      "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                      to_str(expand(obj, tokens, visited))]
                            if len(to_add) > 2:
                                svos.append(to_add)
                            to_add = [to_str(expand(sub, tokens, visited)),
                                      "!" + v2.lower_ if verbNegated or objNegated else v2.lower_,
                                      to_str(expand(obj, tokens, visited))]
                            if len(to_add) > 2:
                                svos.append(to_add)

            else:
                v, objs = _get_all_objs(v, is_pas)
                for sub in subs:
                    if len(objs) > 0:
                        for obj in objs:
                            objNegated = _is_negated(obj)
                            if is_pas:  # reverse object / subject for passive
                                to_add = [to_str(expand(obj, tokens, visited)),
                                          "!" + v.lemma_ if verbNegated or objNegated else v.lemma_,
                                          to_str(expand(sub, tokens, visited))]
                                if len(to_add) > 2:
                                    svos.append(to_add)
                            else:
                                to_add = [to_str(expand(sub, tokens, visited)),
                                          "!" + v.lower_ if verbNegated or objNegated else v.lower_,
                                          to_str(expand(obj, tokens, visited))]
                                if len(to_add) > 2:
                                    svos.append(to_add)
                    else:
                        # no obj - just return the SV parts
                        to_add = [to_str(expand(sub, tokens, visited)),
                                  "!" + v.lower_ if verbNegated else v.lower_, ]
                        if len(to_add) > 2:
                            svos.append(to_add)

    return svos

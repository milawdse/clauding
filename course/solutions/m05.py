"""Reference solutions — Module 5, production systems and explanation."""

from csai import fol
from csai.fol import parse, term_str


def match_one(goal, facts):
    out = []
    for fact in facts:
        subst = fol.unify(goal, fact)
        if subst is not None:
            out.append(subst)
    return out


def match_body(goals, facts, subst=None):
    subst = {} if subst is None else subst
    if not goals:
        return [subst]
    head, rest = goals[0], tuple(goals[1:])
    bound = fol.substitute(head, subst)
    results = []
    for fact in facts:
        unified = fol.unify(bound, fact, subst)
        if unified is None:
            continue
        results.extend(match_body(rest, facts, unified))
    return results


def fire_once(rules, facts):
    known = list(facts)
    new = []
    for rule in rules:
        for subst in match_body(rule.body, known):
            head = fol.substitute(rule.head, subst)
            if fol.is_ground(head) and head not in known and head not in new:
                new.append(head)
    return new


def saturate(rules, facts, max_rounds=50):
    known = list(facts)
    for _ in range(max_rounds):
        new = fire_once(rules, known)
        if not new:
            break
        known.extend(new)
    return known


def derive_with_reasons(rules, facts):
    reasons = {f: (None, ()) for f in facts}
    known = list(reasons)
    for _ in range(50):
        added = False
        for rule in rules:
            for subst in match_body(rule.body, known):
                head = fol.substitute(rule.head, subst)
                if not fol.is_ground(head) or head in reasons:
                    continue
                premises = tuple(fol.substitute(g, subst) for g in rule.body)
                reasons[head] = (rule, premises)
                known.append(head)
                added = True
        if not added:
            break
    return reasons


def supports(fact, reasons):
    if fact not in reasons:
        return set()
    rule, premises = reasons[fact]
    if rule is None:
        return {fact}
    out = set()
    for p in premises:
        out |= supports(p, reasons)
    return out


# --- project ---------------------------------------------------------------

class ExpertSystem:
    def __init__(self, rules):
        self.rules = list(rules)
        self.facts = []
        self.reasons = {}

    def run(self, facts):
        self.reasons = derive_with_reasons(self.rules, list(facts))
        given = list(facts)
        derived = [f for f in self.reasons if f not in set(given)]
        self.facts = given + derived
        return self.facts

    def how(self, fact, indent=0):
        pad = "  " * indent
        if fact not in self.reasons:
            return f"{pad}{term_str(fact)}   (not derived)"
        rule, premises = self.reasons[fact]
        if rule is None:
            return f"{pad}{term_str(fact)}   (given)"
        lines = [f"{pad}{term_str(fact)}   by {rule}"]
        for p in premises:
            lines.append(self.how(p, indent + 1))
        return "\n".join(lines)

    def consequences(self, fact):
        out = set()
        frontier = [fact]
        while frontier:
            current = frontier.pop()
            for other, (rule, premises) in self.reasons.items():
                if rule is not None and current in premises and other not in out:
                    out.add(other)
                    frontier.append(other)
        return out

    def essential(self, fact):
        given = [f for f, (rule, _) in self.reasons.items() if rule is None]
        needed = set()
        for candidate in given:
            without = [f for f in given if f != candidate]
            if fact not in derive_with_reasons(self.rules, without):
                needed.add(candidate)
        return needed

    def diagnose(self, facts, predicates):
        self.run(facts)
        out = {}
        for f in self.facts:
            if fol.is_compound(f) and f[0] in predicates and len(f) == 2:
                out[f[1]] = f[0]
        return out

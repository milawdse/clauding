"""Reference solutions — Module 7, adversarial search and MCTS."""

import math
import random


def minimax_value(game, state):
    if game.is_terminal(state):
        return game.utility(state)
    values = [minimax_value(game, game.result(state, a))
              for a in game.actions(state)]
    if not values:
        return game.utility(state)
    return max(values) if game.player(state) == game.players[0] else min(values)


def best_move(game, state):
    actions = sorted(game.actions(state))
    if not actions:
        return None
    scored = [(minimax_value(game, game.result(state, a)), a) for a in actions]
    if game.player(state) == game.players[0]:
        best = max(v for v, _ in scored)
    else:
        best = min(v for v, _ in scored)
    return min(a for v, a in scored if v == best)


def alphabeta_value(game, state, alpha=-math.inf, beta=math.inf, counter=None):
    if counter is not None:
        counter["nodes"] = counter.get("nodes", 0) + 1
    if game.is_terminal(state):
        return game.utility(state)
    if game.player(state) == game.players[0]:
        value = -math.inf
        for action in game.actions(state):
            value = max(value, alphabeta_value(
                game, game.result(state, action), alpha, beta, counter))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = math.inf
    for action in game.actions(state):
        value = min(value, alphabeta_value(
            game, game.result(state, action), alpha, beta, counter))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def uct_score(total_reward, visits, parent_visits, c=1.4):
    if visits == 0:
        return math.inf
    return total_reward / visits + c * math.sqrt(math.log(parent_visits) / visits)


def rollout(problem, state, rng, max_depth=50):
    for _ in range(max_depth):
        if problem.is_terminal(state):
            break
        options = list(problem.actions(state))
        if not options:
            break
        state = problem.result(state, rng.choice(options))
    return problem.reward(state) if problem.is_terminal(state) else 0.0


def backpropagate(node, reward):
    count = 0
    while node is not None:
        node.visits += 1
        node.total += reward
        count += 1
        node = node.parent
    return count


# --- project ---------------------------------------------------------------

def tree_search(problem, iterations, c=1.4, rng=None, evaluate=None):
    rng = rng or random.Random(0)
    root = Node(problem.initial())
    root.untried = list(problem.actions(root.state))
    best_reward, best_state = 0.0, None
    used = 0

    for i in range(iterations):
        used = i + 1
        node = root
        # 1. select
        while not node.untried and node.children and not problem.is_terminal(node.state):
            node = max(node.children.values(),
                       key=lambda ch: uct_score(ch.total, ch.visits,
                                                node.visits, c))
        # 2. expand
        if node.untried and not problem.is_terminal(node.state):
            action = node.untried.pop(rng.randrange(len(node.untried)))
            child_state = problem.result(node.state, action)
            child = Node(child_state, parent=node, action=action)
            child.untried = list(problem.actions(child_state))
            node.children[action] = child
            node = child
        # 3. simulate
        if evaluate is None:
            state = node.state
            for _ in range(50):
                if problem.is_terminal(state):
                    break
                options = list(problem.actions(state))
                if not options:
                    break
                state = problem.result(state, rng.choice(options))
        else:
            state = node.state
            for _ in range(50):
                if problem.is_terminal(state):
                    break
                options = list(problem.actions(state))
                if not options:
                    break
                scored = [(evaluate(problem.result(state, a)), rng.random(), a)
                          for a in options]
                state = problem.result(state, max(scored)[2])
        reward = problem.reward(state) if problem.is_terminal(state) else 0.0
        if problem.is_terminal(state) and reward > best_reward:
            best_reward, best_state = reward, state
        # 4. backpropagate
        backpropagate(node, reward)
        if best_reward >= 1.0:
            break

    return {
        "best_reward": best_reward,
        "best_state": best_state,
        "iterations": used,
        "root": root,
        "solved": best_reward >= 1.0,
    }

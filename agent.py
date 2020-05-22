import json
import pathlib
import sys
import time

import numpy as np
from scipy.special import softmax
from sortedcontainers import SortedSet

from src.engine import State, Action, ActionType

base_path = str(pathlib.Path(__file__).parent.absolute())

PATH_FIRST_MODEL = "models/1st.json"
PATH_SECOND_MODEL = "models/2nd.json"


def read_game_input():
    # read players info
    game_input = [input(), input()]

    # read cards in hand and actions from opponent
    opp_hand, opp_actions = [int(i) for i in input().split()]
    game_input.append(f"{opp_hand} {opp_actions}")

    # read all opponent actions
    for i in range(opp_actions):
        game_input.append(input())  # opp action #i

    # read card count
    card_count = int(input())
    game_input.append(str(card_count))

    # read cards
    for i in range(card_count):
        game_input.append(input())  # card #i

    return game_input


def encode_state(game_input):
    # initialize empty state
    state = np.zeros((3, 16), dtype=np.float32)

    # get how many opponent action lines to skip
    opp_actions = int(game_input[2].split()[1])

    # put choices from player hand into the state
    for i, card in enumerate(game_input[4 + opp_actions:]):
        card = card.split()

        card_type = [1.0 if int(card[3]) == i else 0.0 for i in range(4)]
        cost = int(card[4]) / 12
        attack = int(card[5]) / 12
        defense = max(-12, int(card[6])) / 12
        keywords = list(map(int, map(card[7].__contains__, 'BCDGLW')))
        player_hp = int(card[8]) / 12
        enemy_hp = int(card[9]) / 12
        card_draw = int(card[10]) / 2

        state[i] = card_type + [cost, attack, defense, player_hp,
                                enemy_hp, card_draw] + keywords

    return state.flatten()


def eval_creature(creature):
    return 3 * creature.attack + 2 * creature.defense \
           + creature.attack * int(creature.has_ability('W')) \
           + 25 * int(creature.has_ability('G')) \
           + 30 * int(creature.has_ability('L'))


def eval_state(state):
    score = 0

    pl = state.current_player
    op = state.opposing_player

    # check opponent's death
    if op.health <= 0:
        score += 100000

    # check own death
    if pl.health <= 0:
        score -= 100000

    # health difference
    score += (pl.health - op.health) * 5

    # card strength
    for pl_lane, op_lane in zip(pl.lanes, op.lanes):
        score += sum(eval_creature(c) for c in pl_lane)
        score -= sum(eval_creature(c) for c in op_lane)

    state._score = score

    return score


def act_on_battle(state):
    # start timer
    start_time = time.process_time()

    # initialize score dict
    # maximize state score primarily and minimize amount of actions secondarily
    scores = dict({(): (eval_state(state), 0)})

    # initialize open and closed sets
    unvisited = SortedSet([()], key=scores.__getitem__)
    visited = SortedSet(key=scores.__getitem__)

    # while there are nodes unvisited
    while unvisited:
        # roll back to starting state
        state.undo_all()

        # get best unvisited node
        actions = unvisited.pop()

        # roll out actions to get into the intended state
        for action in actions:
            state.act(action)

        # and mark it visited
        visited.add(actions)

        # discover all neighbors
        for action in state.available_actions:
            # do the respective action
            state.act(action)

            # calculate score and negative amount of actions
            scores[(*actions, action)] = eval_state(state), - (len(actions) + 1)

            # add this neighbor to the unvisited set
            unvisited.add((*actions, action))

            # calculate time elapsed
            time_elapsed = time.process_time() - start_time

            # if we reached 160 ms, stop the search
            if time_elapsed >= 0.16:
                # consider all unexplored as explored
                visited.update(unvisited)

                # return the actions needed to reach the best node we saw
                best_actions = visited[-1]

                # if any attack on the opponent is available, why not?
                for remaining_action in state.available_actions:
                    if remaining_action.type == ActionType.ATTACK \
                            and remaining_action.target is None:
                        best_actions = (*best_actions, remaining_action)

                # print total elapsed time to stderr
                print("%.3f ms*" % (time_elapsed * 1000), file=sys.stderr)

                return best_actions

            # roll back action
            state.undo()

    # recalculate elapsed time just in case
    time_elapsed = time.process_time() - start_time

    # print total elapsed time to stderr
    print("%.3f ms" % (time_elapsed * 1000), file=sys.stderr)

    # return the actions needed to reach the best node we saw
    return visited[-1]


def act_on_draft(network, state):
    i = 0

    # do a forward pass through all fully connected layers
    while f"model/shared_fc{i}/w:0" in network:
        weights = network[f"model/shared_fc{i}/w:0"]
        biases = network[f"model/shared_fc{i}/b:0"]

        state = np.dot(state, weights) + biases
        state = network['act_fun'](state)

        i += 1

    # calculate the policy
    pi = np.dot(state, network["model/pi/w:0"]) + network["model/pi/b:0"]
    pi = softmax(pi)

    # extract the deterministic action
    action = np.argmax(pi)

    return action


def load_model(path: str):
    # read the parameters
    with open(base_path + "/" + path, "r") as json_file:
        params = json.load(json_file)

    network = dict((label, np.array(weights)) for label, weights in params.items())

    network["act_fun"] = dict(
        tanh=np.tanh,
        relu=lambda x: np.maximum(x, 0),
        elu=lambda x: np.where(x > 0, x, np.exp(x) - 1)
    )[params["act_fun"]]

    return network


def run():
    network = None

    while True:
        game_input = read_game_input()

        # if mana is zero then it is draft phase
        is_draft_phase = int(game_input[0].split()[1]) == 0

        # if network was not loaded, load it
        if network is None:
            playing_first = game_input[0].split()[2] == game_input[1].split()[2]

            path = PATH_FIRST_MODEL if playing_first else PATH_SECOND_MODEL

            network = load_model(path)

        if is_draft_phase:
            state = encode_state(game_input)
            action = act_on_draft(network, state)

            print("PICK", action)
        else:
            state = State.from_native_input(game_input)
            actions = act_on_battle(state)

            if actions:
                print(";".join(map(Action.to_native, actions)))
            else:
                print("PASS")


if __name__ == '__main__':
    run()

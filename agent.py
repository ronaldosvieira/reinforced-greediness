import json
import pathlib
from typing import List

import numpy as np
from scipy.special import softmax

from src.engine import State, ActionType, Action, Phase

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


def eval_state(state):
    score = 0

    pl = state.current_player
    op = state.opposing_player

    # check opponent's death
    if op.health <= 0:
        score += 1000

    # check own death
    if pl.health <= 0:
        score -= 1000

    # health difference
    score += (pl.health - op.health) * 2

    for pl_lane, op_lane in zip(pl.lanes, op.lanes):
        # card count
        score += (len(pl_lane) - len(pl_lane)) * 10

        # card strength
        score += sum(c.attack + c.defense for c in pl_lane)
        score -= sum(c.attack + c.defense for c in op_lane)

    return score


def act_on_battle(state) -> List[Action]:
    actions = []

    while state.winner is None:
        action = act_greedly(state)

        if action.type == ActionType.PASS:
            break

        state.act(action)
        actions.append(action)

    return actions


def act_greedly(state) -> Action:
    best_action, best_score = None, float("-inf")

    for action in state.available_actions:
        state_copy = state.clone()
        state_copy.act(action)

        score = eval_state(state_copy)

        if score > best_score:
            best_action, best_score = action, score

    return best_action


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
            state = State.from_str(game_input)
            actions = act_on_battle(state)

            if actions:
                print(";".join(map(Action.to_native, actions)))
            else:
                print("PASS")


if __name__ == '__main__':
    run()

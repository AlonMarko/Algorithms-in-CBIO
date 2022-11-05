import argparse
import numpy as np
import pandas as pd
from scipy.special import logsumexp

np.seterr(divide='ignore')

letters_and_symbols = {"A": 0, "C": 1, "G": 2, "T": 3, "$": 4, "^": 5}
end_emissions_p = {"A": [0], "C": [0], "G": [0], "T": [0], "$": [1], "^": [0]}
begin_emissions_p = {"A": [0], "C": [0], "G": [0], "T": [0], "$": [0], "^": [1]}
back_ground_emissions_p = {"A": [0.25], "C": [0.25], "G": [0.25], "T": [0.25], "$": [0], "^": [0]}


def transition_emission_creator(path, p, q):
    """
    creates the emission table of motif_len + 4 by 6.
    :param q: the probability of transition from a background state (B1, B2) to the
    next state (M1, Bend).
    :param p: he probability of transition from Bstart to B1.
    :param path: the path to the initial_emissions.tsv
    :return:
    """
    initial_emission = (pd.read_csv(path, sep="\t"))
    motif_len = initial_emission.shape[0]
    initial_emission["$"] = [float(0)] * len(initial_emission.index)
    initial_emission["^"] = [float(0)] * len(initial_emission.index)
    begin_emissions = pd.DataFrame(begin_emissions_p)
    end_emissions = pd.DataFrame(end_emissions_p)
    background_emissions = pd.DataFrame(back_ground_emissions_p)
    emissions_table = pd.concat(
        (begin_emissions, background_emissions, initial_emission, background_emissions, end_emissions), axis=0)
    state_lst = create_states_lst(motif_len)
    emissions_table = emissions_table.set_axis(state_lst, axis=0)
    transition_table = create_transition_table(emissions_table, p, q, motif_len)
    return np.array(emissions_table), motif_len, np.array(transition_table)


def create_states_lst(motif_len):
    """
    creates a list of states for the dataframe construction
    :param motif_len:  the motif length recieved from the initial emissions
    :return: a list of states (strings)
    """
    states = ["Bstart", "B1"]
    for i in range(1, motif_len + 1):
        states.append("M" + str(i))
    states += ["B2", "Bend"]
    return states


def print_func(states, seq, motif_len):
    """
    prints the results of viterbi and posterior
    :param motif_len: motif length
    :param states: the states we got from posterior or viterbi encoding
    :param seq: the input sequence
    :return: prints on screen according to the format given.
    """
    states_lst = ["M" if 2 <= i < motif_len + 2 else "B" for i in states]
    states_str = "".join(states_lst)
    for i in range(0, len(seq), 50):
        line_end = min(i + 50, len(seq))
        print(states_str[i:line_end])
        print(seq[i:line_end])
        print()


def create_transition_table(emission_table, p, q, motif_len):
    """
    :param emission_table:
    :param p:
    :param q:
    :param motif_len:
    :return:
    """
    T = pd.DataFrame(np.zeros((emission_table.shape[0], emission_table.shape[0])), index=emission_table.index,
                     columns=emission_table.index)
    T['B1']['Bstart'] = q
    T['B2']['Bstart'] = 1 - q
    T['B1']['B1'] = 1 - p
    T['B2']['B2'] = 1 - p
    T['Bend']['B2'] = p
    T['M1']['B1'] = p
    for i in range(1, motif_len):
        T[f'M{i + 1}'][f'M{i}'] = 1
    T['B2'][f'M{motif_len}'] = 1
    T['Bend']['Bend'] = 1
    return T


def forward(seq, emission_table, transition_table, motif_len):
    """
    creates the forward matrix
    :param seq: the given sequence + our control additions
    :param emission_table: the emission table we created
    :param transition_table: the trainsition table we created
    :param motif_len: the motif length
    :return: the forward algorithm table of shape states X sequence
    """
    F = np.log(np.zeros((motif_len + 4, len(seq)), dtype=float))
    T = np.log(transition_table)
    E = np.log(emission_table)
    F[0][0] = np.log(1)
    for col in range(1, len(seq)):
        for row in range(motif_len + 4):
            log_sum = logsumexp(F.T[col - 1] + T.T[row])
            F[row][col] = log_sum + E[row][letters_and_symbols[seq[col]]]
    return F


def backward(seq, emission_table, transition_table, motif_len):
    """
    creates the backward matrix
    :param seq: the sequence + our control addons
    :param emission_table: the created emission table
    :param transition_table: the created transition table for the states
    :param motif_len: the motif length
    :return: backward matrix of shape states X sequence
    """
    B = np.log(np.ones((motif_len + 4, len(seq)), dtype=float))
    T = np.log(transition_table)
    E = np.log(emission_table)
    for col in range(len(seq) - 2, -1, -1):
        for row in range(motif_len + 4):
            B[row][col] = logsumexp(B.T[col + 1] + T[row] + E.T[letters_and_symbols[seq[col + 1]]])
    return B


def posterior(seq, emission_table, transition_table, motif_len):
    """
    posterior decoding algorithm - calculates the probability for each state for the ith' position
    in the sequence.
    :param seq: the sequence
    :param emission_table: the emission table
    :param transition_table: the transition table
    :param motif_len: the motif length
    :return: the probability for each state for each position in the sequence.
    """
    B = backward(seq, emission_table, transition_table, motif_len)
    F = forward(seq, emission_table, transition_table, motif_len)
    P = F + B  # log space
    return np.argmax(P, axis=0)


def viterbi(seq, emission_table, transition_table, motif_len):
    """
    viterbi decoding - calculates the probability for each state for each position in the sequence
    :param seq: the sequence
    :param emission_table: the emission table
    :param transition_table: the transition table
    :param motif_len: the motif length
    :return: the probability for each state for each position in the sequence
    """
    V = np.log(np.zeros((motif_len + 4, len(seq)), dtype=float))
    Ptr = np.zeros((motif_len + 4, len(seq)), dtype=int)
    T = np.log(transition_table)
    E = np.log(emission_table)
    V[0][0] = 0
    for col in range(1, len(seq)):
        for row in range(motif_len + 4):
            V[row][col] = E[row][letters_and_symbols[seq[col]]] + np.max(
                V.T[col - 1] + T.T[row])
            Ptr[row][col] = np.argmax(V.T[col - 1] + T.T[row])

    # traceback
    states = []
    state = motif_len + 3
    for i in range(len(seq) - 1, -1, -1):
        states.append(state)
        state = Ptr[state][i]
    return states[::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()
    seq = "^" + args.seq + "$"
    p = args.p
    q = args.q
    emission_table, motif_len, transition_table = transition_emission_creator(args.initial_emission, p, q)
    if args.alg == 'viterbi':
        states = viterbi(seq, emission_table, transition_table, motif_len)
        print_func(states[1:len(states) - 1], seq[1:len(seq) - 1], motif_len)

    elif args.alg == 'forward':
        pass
        F = forward(seq, emission_table, transition_table, motif_len)
        print(F[motif_len + 3][len(seq) - 1])

    elif args.alg == 'backward':
        B = backward(seq, emission_table, transition_table, motif_len)
        print(B[0][0])

    elif args.alg == 'posterior':
        states = posterior(seq, emission_table, transition_table, motif_len)
        print_func(states[1:len(states) - 1], seq[1:len(seq) - 1], motif_len)


if __name__ == '__main__':
    main()

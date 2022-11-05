from motif_find import *
import numpy as np
from Bio.SeqIO.FastaIO import SimpleFastaParser


def parse_args():
    """
    Parse the command line arguments.
    :return: The parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta', help='File path with list of sequences (e.g. seqs_ATTA.fasta)')
    parser.add_argument('seed', help='Guess for the motif (e.g. ATTA)')
    parser.add_argument('p', type=float, help='Initial guess for the p transition probability (e.g. 0.01)')
    parser.add_argument('q', type=float, help='Initial guess for the q transition probability (e.g. 0.9)')
    parser.add_argument('alpha', type=float, help='Softening parameter for the initial profile (e.g. 0.1)')
    parser.add_argument('convergenceThr', type=float, help='ll improvement threshold for the stopping condition'
                                                           ' (e.g. 0.1)')
    return parser.parse_args()


def build_emission(motif, alpha):
    initial_emission = np.full((len(motif), 4), alpha)
    for i in range(len(motif)):
        initial_emission[i][letters_and_symbols[motif[i]]] = 1 - 3 * alpha
    return initial_emission


def extract_seqs(fasta_path):
    fasta_seqs = []
    with open(fasta_path, 'r') as file:
        for name, seq in SimpleFastaParser(file):
            fasta_seqs.append(seq)
    return fasta_seqs


def e_step(motif_len, seq, F, B, curr_seq_ll, curr_emission, curr_transition, E, T):
    for k in range(motif_len):
        for letter in range(len(seq)):
            emission_val = F[k + 2][letter] + B[k + 2][letter] - curr_seq_ll
            curr_emission[k][letters_and_symbols[seq[letter]]] = logsumexp(
                [curr_emission[k][letters_and_symbols[seq[letter]]], emission_val])

    for k in range(4):
        for j in range(motif_len + 4):  # j = l
            for i in range(1, len(seq)):
                if k > 2:
                    transition_val = F[motif_len + 2][i - 1] + T[motif_len + 2][j] + \
                                     E[j][letters_and_symbols[seq[i]]] + B[j][i] - curr_seq_ll
                else:
                    transition_val = F[k][i - 1] + T[k][j] + E[j][letters_and_symbols[seq[i]]] + B[j][i] - curr_seq_ll
                curr_transition[k][j] = logsumexp([curr_transition[k][j], transition_val])

    return curr_emission, curr_transition


def m_step(curr_emission, curr_transition, motif_len):
    transition = np.exp(curr_transition)
    transition_sums = np.sum(transition, axis=1)
    p = (transition[1][2] + transition[3][motif_len + 3]) / (
            transition_sums[1] + transition_sums[3])
    transition = np.divide(transition.T, transition_sums).T
    q = transition[0][1]

    emission = np.exp(curr_emission)
    emission = np.divide(emission.T, np.sum(emission, axis=1)).T
    return emission, p, q


def EM_algorithm(seqs, p, q, threshold, initial_emission):
    ll_history = open('ll_history.txt', 'w+')
    ll_curr = 0

    while True:
        emission, motif_len, transition = transition_emission_creator(initial_emission, p, q)
        ll_prev = ll_curr
        ll_curr = 0
        curr_emission = np.log(np.zeros((motif_len, len(letters_and_symbols)), dtype=float))
        curr_transition = np.log(np.zeros((4, motif_len + 4), dtype=float))
        E = np.log(emission)
        T = np.log(transition)

        for seq in seqs:
            seq = '^' + seq + '$'
            F = forward(seq, emission, transition, motif_len)
            B = backward(seq, emission, transition, motif_len)
            curr_seq_ll = F[motif_len + 3][len(seq) - 1]
            ll_curr += curr_seq_ll

            curr_emission, curr_transition = e_step(motif_len, seq, F, B, curr_seq_ll, curr_emission, curr_transition,
                                                    E, T)
        emission[2:motif_len + 2], p, q = m_step(curr_emission, curr_transition, motif_len)

        initial_emission = emission[2:motif_len + 2, :4]
        ll_history.write(f'{ll_curr}\n')
        if ll_prev != 0 and threshold > ll_curr - ll_prev:
            ll_history.close()
            return emission[2:motif_len + 2, :4], p, q


def output_to_files(initial_emission, p, q, seqs):
    np.savetxt('motif_profile.txt', initial_emission.T, delimiter='\t', fmt="%.2f")
    with open('motif_profile.txt', 'a') as f:
        f.write(f'{round(q, 4)}\n{round(p, 4)}')

    emission, motif_len, transition = transition_emission_creator(initial_emission, p, q)
    with open('motif_positions.txt', 'w+') as f:
        for i, seq in enumerate(seqs):
            position = viterbi(f'^{seq}$', emission, transition, motif_len)
            try:
                index = position.index(2) - 1
            except ValueError:
                index = -1
            f.write(f'{index}\n')


def main():
    args = parse_args()
    fasta_path = args.fasta
    motif = args.seed
    p = args.p
    q = args.q
    alpha = args.alpha
    threshold = args.convergenceThr

    initial_emission = build_emission(motif, alpha)
    fasta_seqs = extract_seqs(fasta_path)

    initial_emission, p, q = EM_algorithm(fasta_seqs, p, q, threshold, initial_emission)
    output_to_files(initial_emission, p, q, fasta_seqs)


if __name__ == "__main__":
    main()

import argparse
import numpy as np
from itertools import groupby
import pandas as pd

DIAGONAL = 0
UP = 1
LEFT = 2
NEW_ALIGN = 3
GAP = '-'


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def printAlignment(a_align, b_align, align_type, best_score):
    """
    Prints the two given aligned sequences in the required format
    :param a_align: the first sequence to print
    :param b_align: the second sequence to print
    :param align_type: the type of alignment the sequences aligned according to
    :param best_score: the score for the best alignment of the given type with the given score matrix
    :return: None
    """
    for i in range(0, len(a_align), 50):
        line_end = min(i + 50, len(a_align))
        print(a_align[i:line_end])
        print(b_align[i:line_end] + '\n')
    print(f'{align_type} : {str(int(best_score))}')


def getMax(T, len_a, len_b, align_type):
    """
    Find the maximum score for the given alignment type and return the score and the indices where it found
    :param T: the scoring table
    :param len_a: the length of one sequence (representing the number of lines)
    :param len_b: the length of other sequence (representing the number of columns)
    :param align_type: the type of alignment applied to create the table T
    :return: the maximal score for the given alignment type and the indices the it can be found (float and ints)
    """
    if align_type == 'global':
        return T[len_a, len_b], len_a, len_b
    elif align_type == 'overlap':
        return np.max(T[len_a, :]), len_a, np.argmax(T[len_a, :])
    max_score = np.max(T)
    indices = np.where(T == max_score)
    return max_score, indices[0][0], indices[1][0]


def align(seq_a, seq_b, score_matrix, align_type):
    """
    Gather all alignment stages and pass the right argument from one to the other
    :param seq_a: one of the sequences to align
    :param seq_b: the other sequence to align
    :param score_matrix: the scores for the alignment
    :param align_type: the type of alignment to perform
    :return: the aligned sequences and the score it got (strings and float)
    """
    len_a = len(seq_a)
    len_b = len(seq_b)
    T, V, scores = initTables(align_type, len_a, len_b, score_matrix, seq_a, seq_b)
    fillTables(T, V, len_a, len_b, score_matrix, scores, seq_a, seq_b)
    max_score, i, j = getMax(T, len_a, len_b, align_type)
    a_align, b_align = extractAlignment(V, i, j, seq_a, seq_b, align_type)
    return a_align, b_align, max_score


def extractAlignment(V, len_a, len_b, seq_a, seq_b, align_type):
    """
    Extract the aligned sequences from the table V
    :param V: the directions table
    :param len_a: the length of the part of seq_a to align (not necessarily equals len(seq_a))
    :param len_b: the length of the part of seq_b to align (not necessarily equals len(seq_b))
    :param seq_a: one of the sequences to extract the alignment for
    :param seq_b: the other sequence to extract the alignment for
    :param align_type: the type of alignment the sequences aligned according to
    :return: two aligned sequences (strings)
    """
    i = len_a
    j = len_b
    a_align = ''
    b_align = ''
    while i > 0 or j > 0:
        if V[i, j] == DIAGONAL:
            a_align = seq_a[i - 1] + a_align
            b_align = seq_b[j - 1] + b_align
            i -= 1
            j -= 1
        elif V[i, j] == UP:
            a_align = seq_a[i - 1] + a_align
            b_align = GAP + b_align
            i -= 1
        elif V[i, j] == LEFT:
            a_align = GAP + a_align
            b_align = seq_b[j - 1] + b_align
            j -= 1
        else:
            break

    if align_type == 'overlap' and len_b < len(seq_b):
        a_align = a_align + GAP * (len(seq_b) - len_b)
        b_align = b_align + seq_b[len_b:]

    return a_align, b_align


def fillTables(T, V, len_a, len_b, score_matrix, scores, seq_a, seq_b):
    """
    Fill the tables T and V according to the requirements
    :param T: the scoring table
    :param V: the direction table
    :param len_a: the length of seq_a
    :param len_b: the length of seq_b
    :param score_matrix: the score matrix determine the score to be given
    :param scores: a vector to assign values for and use as an helper
    :param seq_a: one of the sequences to fill the tables according to
    :param seq_b: the other sequence to fill the tables according to
    :return: None
    """
    for i in range(len_a):
        for j in range(len_b):
            scores[0] = T[i, j] + score_matrix[seq_a[i]][seq_b[j]]
            scores[1] = T[i, j + 1] + score_matrix[seq_a[i]][GAP]
            scores[2] = T[i + 1, j] + score_matrix[GAP][seq_b[j]]
            max_t = np.argmax(scores)
            T[i + 1, j + 1] = scores[max_t]
            V[i + 1, j + 1] = max_t


def initTables(align_type, len_a, len_b, score_matrix, seq_a, seq_b):
    """
    Initiate the tables before the recursive filling
    :param align_type: the type of alignment to follows the rules of
    :param len_a: the length of seq_a
    :param len_b: the length of seq_b
    :param score_matrix: the score matrix determine the score to be given
    :param seq_a: one of the sequences to initialize the tables according to
    :param seq_b: the other sequence to initialize the tables according to
    :return: the scoring and direction tables and a scores vector ((len_a + 1)X(len_b + 1) two dimensional numpy arrays
    and  either 3 or 4 cells long one dimensional numpy array)
    """
    T = np.zeros((len_a + 1, len_b + 1))
    V = np.zeros((len_a + 1, len_b + 1))

    for i in range(len_a):
        T[i + 1, 0] = score_matrix[GAP][seq_a[i]] + T[i, 0] if align_type == 'global' else 0
    V[:, 0] = NEW_ALIGN if align_type == 'local' else UP

    for j in range(len_b):
        T[0, j + 1] = 0 if align_type == 'local' else score_matrix[GAP][seq_b[j]] + T[0, j]
    V[0, :] = NEW_ALIGN if align_type == 'local' else LEFT

    scores = np.zeros(4) if align_type == 'local' else np.zeros(3)
    return T, V, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)', required=True)
    parser.add_argument('--score', help='Score matrix in.tsv format (default is score_matrix.tsv) ',
                        default='score_matrix.tsv')
    command_args = parser.parse_args()

    seq_a = tuple(fastaread(command_args.seq_a))[0][1]
    seq_b = tuple(fastaread(command_args.seq_b))[0][1]
    score_mat = pd.read_csv(command_args.score, sep='\t', index_col=[0])

    if command_args.align_type == 'global':
        a_align, b_align, max_score = align(seq_a, seq_b, score_mat, command_args.align_type)
    elif command_args.align_type == 'local':
        a_align, b_align, max_score = align(seq_a, seq_b, score_mat, command_args.align_type)
    elif command_args.align_type == 'overlap':
        a_align, b_align, max_score = align(seq_a, seq_b, score_mat, command_args.align_type)
    else:
        print('Unknown align_type. Please choose \'global\', \'local\' or \'overlap\'')
        return
    printAlignment(a_align, b_align, command_args.align_type, max_score)


if __name__ == '__main__':
    main()

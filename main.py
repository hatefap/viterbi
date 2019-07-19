from data_util import  Bijenkhan
from viterbi import Viterbi
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score

GREEN = "\033[0;32m"
RED = "\033[1;31m"
CYAN = "\033[1;36m"

BIJEN_CORPUS = '/home/hatef/courses/Term-2/NLP/HWs/CA4/train.txt'
NUM_TEST_SAMPES = 50

def main():
    bijen = Bijenkhan(BIJEN_CORPUS)
    sents_tags = []
    for sents, tags in bijen.sent_tag_gen(100):
        s = zip(sents, tags)
        sents_tags.extend(s)
    random.shuffle(sents_tags)
    test_sents_tags = sents_tags[:NUM_TEST_SAMPES]
    train_sents_tags = sents_tags[NUM_TEST_SAMPES:]
    viterbi = Viterbi(len(bijen.get_tags()),
                      len(bijen.get_vocab()),
                      bijen.get_tags(),
                      bijen.get_bigram_tags(),
                      train_sents_tags)

    for i in range(len(test_sents_tags)):
        true_labels = test_sents_tags[i][1]
        print(GREEN + 'True labels: ', true_labels)
        tmp = test_sents_tags[i][0]
        pred_labels = viterbi.viterbi(tmp[1:-1])
        print(RED + 'Pred labels: ', pred_labels)
        print(CYAN + f'Accuracy: {accuracy_score(true_labels, pred_labels)}')
        print(CYAN + f'Precision: {precision_score(true_labels, pred_labels, average="macro")}')
        print(CYAN + f'Recall: {recall_score(true_labels, pred_labels, average="macro")}')
        print('\n'*2)



if __name__ == '__main__':
    main()
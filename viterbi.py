from typing import List, Tuple
import numpy as np
from nltk.lm import Laplace

class Viterbi():

    def __init__(self, num_of_tags, num_of_vocab, tags, bigram_tags, sents_tags: List[Tuple[List, List]]):
        self.num_of_tags = num_of_tags
        self.num_of_vocab = num_of_vocab
        self.tags = list(tags)
        self.bigram_tags = bigram_tags
        self.sents_tags = sents_tags
        self.laplace = Laplace(2)
        self.laplace.fit(text=[self.bigram_tags], vocabulary_text=self.tags)

        self.cache = {}

    def _get_prob_wi_ti(self, word, tag):
        key = word + tag
        if key in self.cache:
            return self.cache[key]
        tag_cnt = 0
        word_tag_cnt = 0
        for s, t in self.sents_tags:
            if len(s) != len(t):
                raise Exception(f'sentence and tag are not aligned.\n sentence:{s}\n tag:{t}')
            tag_cnt += t.count(tag)
            word_tag_cnt += len([i for i in range(len(s)) if s[i] == word and t[i] == tag])

        self.cache[key] = (word_tag_cnt + 1) / (tag_cnt + self.num_of_vocab)
        return self.cache[key]

    def _get_prob_ti_1_and_ti(self, ti_1: str, ti: str):
        return self.laplace.score(ti, [ti_1])

    def viterbi(self, sentence):
        """
        sentence = ['من','به','مدرسه','رفتم']
        """
        # init viterbi table

        viterbi = [[0 for _ in range(len(sentence) + 1)] for _ in range(self.num_of_tags)]
        backtrace = [[0 for _ in range(len(sentence) + 1)] for _ in range(self.num_of_tags)]

        for index in range(len(self.tags)):
            p_trans = self._get_prob_ti_1_and_ti('<s>', self.tags[index])
            p_emis = self._get_prob_wi_ti(sentence[0], self.tags[index])
            viterbi[index][0] = p_trans * p_emis
            backtrace[index][0] = 0

        for w_index in range(1, len(sentence)):
            cur_word = sentence[w_index]
            for t_index in range(len(self.tags)):
                cur_tag = self.tags[t_index]
                p_emis = self._get_prob_wi_ti(cur_word, cur_tag)
                tmp = [viterbi[i][w_index - 1] * self._get_prob_ti_1_and_ti(self.tags[i], cur_tag) * p_emis for i in range(len(self.tags))]
                viterbi[t_index][w_index] = max(tmp)
                backtrace[t_index][w_index] = np.argmax(
                    [viterbi[i][w_index - 1] * self._get_prob_ti_1_and_ti(self.tags[i], cur_tag) for i in range(len(self.tags))])

        viterbi[-1][-1] = max(
            [viterbi[i][len(sentence) - 1] * self._get_prob_ti_1_and_ti(self.tags[i], '</s>') for i in range(len(self.tags))])
        backtrace[-1][-1] = np.argmax(
            [viterbi[i][len(sentence) - 1] * self._get_prob_ti_1_and_ti(self.tags[i], '</s>') for i in range(len(self.tags))])

        result = backtrace[-1][1:]  # last list contain index of tags
        return ['<s>'] + [self.tags[result[i]] for i in range(0, len(result))] + ['</s>']

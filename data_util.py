import os
from typing import List, Tuple, Set, Iterator
from nltk.util import bigrams


class Bijenkhan():

    def __init__(self, bijenkhan_corpus_loc:str) -> None:
        if os.path.isfile(bijenkhan_corpus_loc):
            self.bijenkhan_corpus_loc = bijenkhan_corpus_loc
        else:
            raise Exception('file not found or it is a directory')

        self.tags = set()
        self.vocab_cnt = 0


    def sent_tag_gen(self, limit: int) -> Tuple[List, List]:
        """
        if you look at Bigenkhan file you see two columns of data
        first column is POS tag, second column is word. this function just aggregate these two columns
        and returns a tuple like this ([[sent_1],[sent_2],[sent_3]...[sent_limit]], [[POS_1],[POS_2][POS_3]...[POS_limit]])
        """
        with open(self.bijenkhan_corpus_loc, 'r') as f:
            eof = False
            while not eof:
                cnt = 0
                sents, tags, cur_sent, cur_tag = [], [], [], []
                while cnt < limit:
                    ln = f.readline().split()
                    if len(ln) > 0:
                        cur_tag.append(ln[-1])
                        cur_sent.append(' '.join([w for w in ln[0:-1]]))
                        # if we reach end of a sentence in corpus
                        if cur_sent[-1] == '.' and cur_tag[-1] == 'DELM':
                            cur_sent.insert(0, '<s>')
                            cur_sent.append('</s>')
                            cur_tag.insert(0, '<s>')
                            cur_tag.append('</s>')
                            sents.append([w for w in cur_sent])
                            tags.append([t for t in cur_tag])
                            cnt += 1
                            # clear current sentence to start new sentence
                            cur_sent.clear(), cur_tag.clear()

                    # if we reach end of file in corpus
                    else:
                        sents.append([w for w in cur_sent])
                        tags.append([t for t in cur_tag])
                        eof = True
                        break
                yield sents, tags


    def get_vocab(self):
        vocab = set()
        if self.vocab_cnt:
            return self.vocab_cnt
        else:
            s_gen = self.sent_tag_gen(100)
            for sents, _ in s_gen:
                [[vocab.add(word) for word in sent]for sent in sents]
            return vocab

# all tags in corpus
    def get_tags(self) -> Set[str]:
        if self.tags:
            return self.tags
        else:
            # 100 sentences and their POS tags
            s_gen = self.sent_tag_gen(100)
            for _, tags in s_gen:
                [[self.tags.add(tag) for tag in tag_seq]for tag_seq in tags]
            return self.tags

# bigram of tags in corpus
    def get_bigram_tags(self) -> Iterator[Tuple[str, str]]:
        "first concatenate all POS tags in all sentences in bijenkhan corpus"
        seq_tags = []
        gen = self.sent_tag_gen(100)
        for _, tags in gen:
            [[seq_tags.append(tag) for tag in tag_seq] for tag_seq in tags]
        return list(bigrams(seq_tags))








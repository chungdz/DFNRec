import json
import pickle

class ModelConfig():
    def __init__(self, root):

        word_dict = json.load(open('{}/word.json'.format(root), 'r', encoding='utf-8'))
        user_dict = pickle.load(open('{}/user_n.pkl'.format(root), 'rb'))
        news_dict = pickle.load(open('{}/news_n.pkl'.format(root), 'rb'))
        
        self.word_num = len(word_dict)
        self.user_num = len(user_dict)
        self.news_num = len(news_dict)
        self.pos_hist_length = 30
        self.neg_hist_length = 30
        self.unclicked_hist_length = 30
        self.max_title_len = 15
        self.word_dim = 100
        self.hidden_size = 100
        self.head_num = 4
        self.dropout = 0.2
        self.input_len = 2 + (1 + self.pos_hist_length + self.neg_hist_length + self.unclicked_hist_length) * self.max_title_len
        return None
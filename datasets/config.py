import json
import pickle

class ModelConfig():
    def __init__(self):

        word_dict = json.load(open('data/word.json', 'r', encoding='utf-8'))
        user_dict = pickle.load(open('data/user.pkl', 'rb'))
        news_dict = pickle.load(open('data/news.pkl', 'rb'))
        
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
        
        return None
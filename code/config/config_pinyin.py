import random


class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"


class Config():
    def __init__(self):
        self.debug_mode = True
        self.lr = 1e-3
        self.dropout = [0, 0] #[0.2,0.2]#t,a
        self.output_dim = 1
        self.text_out = 64 #32
        self.audio_out = 64 #64
        self.batch_size = 64
        self.post_dim = 32
        self.early_stop = 20
        self.hidden_dim = [128,64,64]#[128,16]
        # sims
        self.input_len = {'text_in': 39, 'audio_in': 400,'pinyin':36}
        self.feature_dim = {'text_in': 768, 'audio_in': 33,'pinyin':300}
        self.text_linear=64
        self.audio_linear=64
        self.weight_t = 0.8
        self.weight_a = 0.2
        params = {
            'debug_mode': True,
            'output_dim': 1,
            'input_len': {'text_in': 39, 'audio_in': 400, 'pinyin': 36},
            'feature_dim': {'text_in': 768, 'audio_in': 33, 'pinyin': 300},
            'd_paras': ['text_out', 'audio_out', 'batch_size', 'hidden_dim', 'dropout', 'lr',
                        'text_linear', 'audio_linear','weight_t','weight_a'],
            'text_out': random.choice([32, 64, 128]),
            'audio_out': random.choice([32, 64, 128]),
            'batch_size': random.choice([32, 64]),
            'hidden_dim': random.choice([[128, 16,16], [64, 64,64], [128, 32,32]]),
            'dropout': random.choice([[0, 0], [0.1, 0.1], [0.2, 0.2]]),
            'lr': random.choice([1e-3, 2e-3, 5e-3]),
            'text_linear': random.choice([16, 32, 64]),
            'audio_linear': random.choice([16, 32, 64]),
            'early_stop':20,
            'weight_t': random.choice([0.2, 0.6, 0.4, 0.6, 0.8, 1.0]),
            'weight_a': random.choice([0.2, 0.6, 0.4, 0.6, 0.8, 1.0])
        }
        self.params = Storage(dict(params))

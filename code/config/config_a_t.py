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
        """
        ASR-SER
        """
        # self.debug_mode = False
        # self.lr = 1e-3
        # self.dropout = [0, 0.1] #[0.2,0.2]#t,a
        # self.output_dim = 1
        # self.text_out = 64 #32
        # self.audio_out = 64 #64
        # self.batch_size = 64
        # self.post_dim = 32
        # self.early_stop = 20
        # self.hidden_dim = [128,32]#[128,16]
        # # sims
        # self.input_len = {'text_in': 39, 'audio_in': 400}
        # self.feature_dim = {'text_in': 768, 'audio_in': 33}
        # self.text_linear=64
        # self.audio_linear=32

        """
        EF-LSTM
        """
        # self.debug_mode = False
        # self.lr = 1e-3
        # self.dropout = 0.4
        # self.output_dim = 1
        # self.batch_size = 128
        # self.early_stop = 20
        # self.hidden_dim = 16
        # self.num_layers=2
        # self.input_len = (39,400)
        # self.feature_dim = (768, 33)
        # self.need_align=True
        # self.need_normalized=False

        """
        LF-DNN
        """
        # self.debug_mode = False
        # self.lr = 5e-4
        # self.dropouts = (0.2, 0.2, 0.2)
        # self.output_dim = 1
        # self.batch_size = 32
        # self.early_stop = 20
        # self.hidden_dims = (128, 16)
        # self.input_lens = (39,400)
        # self.feature_dims = (768, 33)
        # self.text_out=32
        # self.post_fusion_dim=32
        # self.need_align=False
        # self.need_normalized=True

        """
        TFN
        """
        # self.debug_mode = False
        # self.lr = 2e-3
        # self.dropouts = (0.2, 0.2, 0.2)
        # self.output_dim = 1
        # self.batch_size = 128
        # self.early_stop = 20
        # self.hidden_dims = (128, 16)
        # self.input_lens = (39,400)
        # self.feature_dims = (768, 33)
        # self.text_out=128
        # self.post_fusion_dim=64
        # self.need_align = False
        # self.need_normalized=True
        """
        LMF
        """
        # self.debug_mode = False
        # self.lr = 1e-3
        # self.dropouts = (0.3, 0.3, 0.5)
        # self.output_dim = 1
        # self.batch_size = 128
        # self.early_stop = 20
        # self.hidden_dims = (128, 16)
        # self.input_lens = (39,400)
        # self.feature_dims = (768, 33)
        # self.rank=4
        # self.use_softmax=False
        # self.factor_lr=5e-4
        # self.weight_decay=1e-3
        # self.need_align = False
        # self.need_normalized=True
        """
               MFN
        """
        # self.debug_mode = False
        # self.lr = 5e-4
        # self.output_dim = 1
        # self.batch_size = 128
        # self.early_stop = 20
        # self.hidden_dim = (128, 16)
        # self.input_len = (39,400)
        # self.feature_dim = (768, 33)
        # self.memsize=64
        # self.windowsize=2
        # self.NN1Config={"drop": 0.2, "shapes": 32}
        # self.NN2Config={"drop": 0.7, "shapes": 128}
        # self.gamma1Config={"drop": 0.7, "shapes": 256}
        # self.gamma2Config={"drop": 0.7, "shapes": 32}
        # self.outConfig={"drop": 0.7, "shapes": 32}
        # self.need_align = True
        # self.need_normalized=True
        """
        Graph-MFN
        """
        #不需要normalize
        self.debug_mode = False
        self.lr = 5e-4
        self.batch_size = 64
        self.early_stop = 8
        self.hidden_dim = (256, 32)
        self.input_len = (39, 400)
        self.feature_dim = (768, 33)
        self.memsize = 64
        self.inner_node_dim=128
        self.NNConfig = {"drop": 0.5, "shapes": 256}
        self.gamma1Config = {"drop": 0.2, "shapes": 32}
        self.gamma2Config = {"drop": 0.7, "shapes": 32}
        self.outConfig = {"drop": 0.2, "shapes": 256}
        self.weight_decay=0.001
        self.need_align = True
        self.need_normalized = False
        params = {
            'debug_mode': True,
            'output_dim': 1,
            'input_len': {'text_in': 39, 'audio_in': 400, 'video_in': 55},
            'feature_dim': {'text_in': 768, 'audio_in': 33, 'video_in': 709},
            'd_paras': ['text_out', 'audio_out', 'batch_size', 'hidden_dim', 'dropout', 'lr',
                        'text_linear', 'audio_linear'],
            'text_out': random.choice([32, 64, 128]),
            'audio_out': random.choice([32, 64, 128]),
            'batch_size': random.choice([32, 64]),
            'hidden_dim': random.choice([[128, 16], [64, 64], [128, 32]]),
            'dropout': random.choice([[0, 0], [0.1, 0.1], [0.2, 0.2]]),
            'lr': random.choice([1e-3, 2e-3, 5e-3]),
            'text_linear': random.choice([16, 32, 64]),
            'audio_linear': random.choice([8, 16, 32, 64]),
            'early_stop':20
        }
        self.params = Storage(dict(params))

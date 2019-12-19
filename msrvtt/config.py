# configuration for MSR-VTT

class Config():
    def __init__(self, embed):
        self._n_w = 300
        self._n_h = 1024
        self._n_f = 128
        self._n_t = 1700 # 1399
        self._n_v = 13796 # 20192
        self._n_z1 = 2048
        self._n_z2 = 1536
        self._embed = embed
        self._lr = 2e-4
        self._train_size = 130260
        self._train_size2 = 6513
        self._val_size = 497
        self._test_size = 2990
        self._epoch = 16
        self._threshold = 16
        self._max_steps = 20
        self._batch_size = 128
        self._keep_prob = 0.5
        self._wd = 0.9455
        self._gamma = 0.7
        self._avglen = 10
        self._we_trainable = False

    @property
    def n_w(self):
        return self._n_w

    @property
    def n_h(self):
        return self._n_h
    
    @property
    def n_f(self):
        return self._n_f
    
    @property
    def n_v(self):
        return self._n_v
    
    @property
    def n_t(self):
        return self._n_t

    @property
    def n_z(self):
        return self._n_z1 + self._n_z2
        # return self._n_z2

    @property
    def n_z1(self):
        return self._n_z1


    @property
    def n_z2(self):
        return self._n_z2

    @property
    def embed(self):
        return self._embed

    @property
    def lr(self):
        return self._lr
    
    @property
    def train_size(self):
        return self._train_size

    @property
    def train_size2(self):
        return self._train_size2
    
    @property
    def val_size(self):
        return self._val_size
    
    @property
    def test_size(self):
        return self._test_size
    
    @property
    def epoch(self):
        return self._epoch
    
    @property
    def max_steps(self):
        return self._max_steps
     
    @property
    def threshold(self):
        return self._threshold
         
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def keep_prob(self):
        return self._keep_prob
   
    @property
    def wd(self):
        return self._wd

    @property
    def gamma(self):
        return self._gamma

    @property
    def avglen(self):
        return self._avglen

    @property
    def we_trainable(self):
        return self._we_trainable

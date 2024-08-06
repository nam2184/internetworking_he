from Pyfhel import PyCtxt, Pyfhel
import numpy as np

def create_he(scheme='ckks', n=8192, scale=2**26, scale_power = 26, n_mults = 6, end_size = 31 ) :

    HE = Pyfhel()
    HE.contextGen(scheme=scheme, n=n, scale=scale, qi_sizes=[end_size]+ [scale_power]*n_mults +[end_size])
    HE.keyGen()
    return HE

class EncryptedNumber :
    def __init__(self, HE_instance, plaintext = None, cipher = None):
        self.HE_instance = HE_instance
        if cipher is None:
            self.cipher = self.HE_instance.encrypt(plaintext)
        else :
            self.cipher = cipher

    def __add__(self, other):
        if isinstance(other, EncryptedNumber):
            e = EncryptedNumber(self.HE_instance, cipher = self.cipher + other.ciphertext())
        elif isinstance(other, PyCtxt):
            e = EncryptedNumber(self.HE_instance, cipher = self.cipher + other)
        elif isinstance(other, np.ndarray):
            raise ValueError("Encrypt the value first, pass in as EncryptedNumber or Pyctxt")
        elif isinstance(other, int):
            raise ValueError("Make np array for better performance")
        else:
            raise TypeError("Unsupported operand type(s) for +: 'EncryptedNumber' and '{}'".format(type(other).__name__))
        return e

    def ciphertext(self):
        return self.cipher
    
    def decrypted(self):
        return self.HE_instance.decrypt(self.cipher)[0]



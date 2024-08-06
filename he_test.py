import logging
from he import EncryptedNumber, create_he
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TestCipher:
    def __init__(self):
        pass

    def test_encrypt(self):
        logging.info("Starting test_encrypt")

        plaintext = np.array([10], dtype=np.int64)
        HE = create_he()
        e = EncryptedNumber(HE_instance=HE, plaintext=plaintext)

        logging.debug(f"Plaintext: {plaintext}")
        logging.debug(f"Decrypted number: {e.decrypted()}")


    def test_operation(self):
        logging.info("Starting test_operation")

        plaintext1 = np.array([10], dtype=np.int64)
        plaintext2 = np.array([12], dtype=np.int64)
        HE = create_he()

        e1 = EncryptedNumber(HE, plaintext=plaintext1)
        e2 = EncryptedNumber(HE, plaintext=plaintext2)
        sum_encrypted = e1 + e2

        expected_sum = plaintext1 + plaintext2
        decrypted_sum = sum_encrypted.decrypted()

        logging.debug(f"Plaintext1: {plaintext1}")
        logging.debug(f"Plaintext2: {plaintext2}")
        logging.debug(f"Expected sum: {expected_sum}")
        logging.debug(f"Decrypted sum: {decrypted_sum}")
 

    def main(self):
        logging.info("Starting tests")
        self.test_encrypt()
        self.test_operation()
        logging.info("Tests completed")

if __name__ == '__main__':
    unittests = TestCipher()
    unittests.main()

    


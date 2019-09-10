# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Utils to encrypt/decrypt data."""
import base64
from Crypto import Random
from Crypto.Cipher import AES


class AESCipher(object):
    """AES Cipher."""

    def __init__(self):
        """Init block size."""
        self.bs = AES.block_size

    def encrypt(self, raw, key):
        """Encrypt a raw string."""
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc, key):
        """Decrypt an encrypted string."""
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return self._unpad(
            cipher.decrypt(enc[AES.block_size:])
        ).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * \
            chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

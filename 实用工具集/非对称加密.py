from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

# 生成RSA密钥对
def generate_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

# 加密数据
def encrypt_data(public_key, data):
    recipient_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    encrypted_data = cipher_rsa.encrypt(data)
    return encrypted_data

# 解密数据
def decrypt_data(private_key, encrypted_data):
    private_key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(private_key)
    data = cipher_rsa.decrypt(encrypted_data)
    return data

# 示例用法
if __name__ == "__main__":
    private_key, public_key = generate_keys()
    print("Private Key:", private_key.decode())
    print("Public Key:", public_key.decode())

    data = b"Hello, this is a test message."
    encrypted_data = encrypt_data(public_key, data)
    print("Encrypted Data:", encrypted_data)

    decrypted_data = decrypt_data(private_key, encrypted_data)
    print("Decrypted Data:", decrypted_data.decode())

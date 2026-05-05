import falcon

def falcon_signature(n, message, identity):
    message = message.encode()
    sk = falcon.SecretKey(n, identity=identity)
    pk = falcon.PublicKey(sk)
    sign = sk.sign(message)

    return pk,sign

def falcon_verify(pk, sign, message):
    message = message.encode()
    verify = pk.verify(message, sign)

    return verify

if __name__ == '__main__':
    # pk, sign = falcon_signature(512, "416416", "11")
    # c = falcon_verify(pk,sign,"416416")
    # print(c)
    pkg = falcon.PKG(512)
    identity = "user@example.com"
    sk = pkg.extract_user_key(identity)

    message = b"Hello, Falcon!"

    # 方案1：不嵌入H(ID)
    sig1 = sk.sign(message, identity)
    valid1 = sk.verify(message, sig1, identity)
    print(f"方案1验证结果: {valid1}")
    traced_id1 = pkg.trace_signature(message, sig1)
    print(f"方案1溯源ID: {traced_id1}")

    # 方案2：嵌入H(ID)
    sig2 = sk.sign(message, identity, id_hash=True)
    valid2 = sk.verify(message, sig2)  # 无需提供identity
    print(f"方案2验证结果: {valid2}")
    traced_id2 = pkg.trace_signature(message, sig2)
    print(f"方案2溯源ID: {traced_id2}")




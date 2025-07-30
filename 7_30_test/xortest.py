if __name__ == "__main__":
    # --- 키 준비 ---
    print("Generating round keys...")
    master_key_hex = "2b7e151628aed2a6abf7158809cf4f3c"
    master_key_bytes = bytes.fromhex(master_key_hex)
    round_keys_words = key_expansion(master_key_bytes)
    roundkeysone = []
    roundkeystwo = []
    for r in range(11):
        start_index = r * 4
        round_key = sum(round_keys_words[start_index:start_index + 4], [])
        round_key_np = np.array(round_key)
        rkone = round_key_np >> 4
        rktwo = round_key_np % 16
        rkone_encoded = zeta**rkone
        rktwo_encoded = zeta**rktwo
        # For AddRoundKey, we need the key encoded and repeated for each slot item.
        roundkeysone.append([val for val in rkone_encoded for _ in range(512)])
        roundkeystwo.append([val for val in rktwo_encoded for _ in range(512)])
    print("Round keys generated.")

    # --- 데이터 준비 ---
    print("Preparing plaintext data...")
    plaintext = np.array([[random.randrange(256) for _ in range(16)] for _ in range(512)], dtype=np.uint8)
    data = plaintext.T.flatten()
    dataup = data >> 4
    datadown = data % 16
    dataup_encoded = zeta**dataup
    datadown_encoded = zeta**datadown
    cryptedup = engine.encrypt(dataup_encoded, public_key)
    crypteddown = engine.encrypt(datadown_encoded, public_key)
    
    round1 = LUTforaddroundkey(cryptedup, roundkeysone[0])
    round2 = LUTforaddroundkey(round1, roundkeysone[0])
    round2 = engine.decrypt(round2, secret_key)
    print(round2[:4094])
    print(dataup_encoded[:4094])

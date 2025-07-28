if __name__ == "__main__":
    # 예제 키 (16진수 문자열)
    master_key_hex = "2b7e151628aed2a6abf7158809cf4f3c"
    # 키를 바이트로 변환
    master_key_bytes = bytes.fromhex(master_key_hex)

    # 키 스케줄 생성
    round_keys_words = key_expansion(master_key_bytes)
    
    # 라운드 키 출력 (각 라운드는 4개의 워드로 구성)
    roundkeysone = []
    roundkeystwo = []
    for r in range(11): # 0부터 10까지 11개 라운드
        start_index = r * 4
        # 4개의 워드를 하나의 16바이트 라운드 키로 합침
        round_key = sum(round_keys_words[start_index : start_index+4], [])
        round_key = np.array(round_key)
        rkone = round_key >> 4
        rktwo = round_key % 16
        rkone = zeta ** rkone
        rktwo = zeta ** rktwo
        rkone = [round_key[i] for i in range(16) for _ in range(512)]
        rktwo = [round_key[i] for i in range(16) for _ in range(512)]
        roundkeysone.append(rkone)
        roundkeystwo.append(rktwo)

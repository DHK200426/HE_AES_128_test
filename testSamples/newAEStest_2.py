from desilofhe import Engine
import numpy as np
import random

AES_SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]

AES_INV_SBOX = [0] * 256
for i, val in enumerate(AES_SBOX):
    AES_INV_SBOX[val] = i

def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits):
    bytes_out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte |= (bits[i + j] << j)
        bytes_out.append(byte)
    return bytes_out

def encrypt_bit(engine, public_key, bit):
    encoded = engine.encode([float(bit)])
    return engine.encrypt(encoded, public_key)

def decrypt_bit(engine, secret_key, enc_bit):
    decoded = engine.decode(engine.decrypt_to_plaintext(enc_bit, secret_key))
    return round(decoded[0])

def xor_bit(engine, a, b):
    result = engine.add(a, b)
    return result

def safe_rescale(engine, ctxt):
    try:
        res = engine.rescale(ctxt)
        return res
    except RuntimeError as e:
        return ctxt 

def multiplyciphertexted(engine, a, b, relinearization_key, conjugation_key, bootstrap_key):
    now = min(a.level, b.level)
    res = engine.multiply(a, b, relinearization_key)
    if now == 0:
        res = engine.bootstrap(res, relinearization_key, conjugation_key, bootstrap_key)
    return res

def and_bit(engine, a, b, relinearization_key, conjugation_key, bootstrap_key):
    a = safe_rescale(engine, a)
    b = safe_rescale(engine, b)
    multiplied = multiplyciphertexted(engine, a, b, relinearization_key, conjugation_key, bootstrap_key)
    relinearized = engine.relinearize(multiplied, relinearization_key)
    relinearized = safe_rescale(engine, relinearized)
    return relinearized

def not_bit(engine, a, public_key):
    one = encrypt_bit(engine, public_key, 1)
    return engine.subtract(one, a)

class SboxLUT:
    def __init__(self, engine, public_key, relinearization_key, conjugation_key, bootstrap_key):
        self.engine = engine
        self.public_key = public_key
        self.relinearization_key = relinearization_key
        self.conjugation_key = conjugation_key
        self.bootstrap_key = bootstrap_key

        self.enc_zero = encrypt_bit(engine, public_key, 0)
        self.enc_one = encrypt_bit(engine, public_key, 1)

        # 16개 상위 4비트별 하위 4비트 LUT
        self.lut_level1 = {}
        for upper in range(16):
            self.lut_level1[upper] = [
                [encrypt_bit(engine, public_key, (AES_SBOX[(upper << 4) + lower] >> i) & 1) for i in range(8)]
                for lower in range(16)
            ]

    def mux(self, cond, a, b):
        return xor_bit(
            self.engine,
            and_bit(self.engine, cond, xor_bit(self.engine, a, b),
                    self.relinearization_key, self.conjugation_key, self.bootstrap_key),
            b
        )

    def bits_equal(self, bits, value):
        res = self.enc_one
        for i, bit in enumerate(bits):
            if (value >> i) & 1:
                res = and_bit(self.engine, res, bit,
                              self.relinearization_key, self.conjugation_key, self.bootstrap_key)
            else:
                res = and_bit(self.engine, res, not_bit(self.engine, bit, self.public_key),
                              self.relinearization_key, self.conjugation_key, self.bootstrap_key)
        return res

    def select_from_table(self, idx_bits, table):
        result = table[0]
        for i in range(1, 16):
            cond = self.bits_equal(idx_bits, i)
            result = [self.mux(cond, table[i][j], result[j]) for j in range(8)]
        return result

    def lookup(self, enc_bits):
        upper_bits = enc_bits[4:]
        lower_bits = enc_bits[:4]

        upper_condition = [self.bits_equal(upper_bits, upper) for upper in range(16)]

        selected_lower_outputs = [self.select_from_table(lower_bits, self.lut_level1[upper]) for upper in range(16)]

        result = [self.enc_zero for _ in range(8)]
        for cond, out_bits in zip(upper_condition, selected_lower_outputs):
            result = [self.mux(cond, out_bit, res_bit) for out_bit, res_bit in zip(out_bits, result)]

        return result


class InvSboxLUT:
    def __init__(self, engine, public_key, relinearization_key, conjugation_key, bootstrap_key):
        self.engine = engine
        self.public_key = public_key
        self.relinearization_key = relinearization_key
        self.conjugation_key = conjugation_key
        self.bootstrap_key = bootstrap_key

        self.enc_zero = encrypt_bit(engine, public_key, 0)
        self.enc_one = encrypt_bit(engine, public_key, 1)

        # 16개 상위 4비트별 하위 4비트 LUT (역 S-box)
        self.lut_level1 = {}
        for upper in range(16):
            self.lut_level1[upper] = [
                [encrypt_bit(engine, public_key, (AES_INV_SBOX[(upper << 4) + lower] >> i) & 1) for i in range(8)]
                for lower in range(16)
            ]

    def mux(self, cond, a, b):
        return xor_bit(
            self.engine,
            and_bit(self.engine, cond, xor_bit(self.engine, a, b),
                    self.relinearization_key, self.conjugation_key, self.bootstrap_key),
            b
        )

    def bits_equal(self, bits, value):
        res = self.enc_one
        for i, bit in enumerate(bits):
            if (value >> i) & 1:
                res = and_bit(self.engine, res, bit,
                              self.relinearization_key, self.conjugation_key, self.bootstrap_key)
            else:
                res = and_bit(self.engine, res, not_bit(self.engine, bit, self.public_key),
                              self.relinearization_key, self.conjugation_key, self.bootstrap_key)
        return res

    def select_from_table(self, idx_bits, table):
        result = table[0]
        for i in range(1, 16):
            cond = self.bits_equal(idx_bits, i)
            result = [self.mux(cond, table[i][j], result[j]) for j in range(8)]
        return result

    def lookup(self, enc_bits):
        upper_bits = enc_bits[4:]
        lower_bits = enc_bits[:4]

        upper_condition = [self.bits_equal(upper_bits, upper) for upper in range(16)]

        selected_lower_outputs = [self.select_from_table(lower_bits, self.lut_level1[upper]) for upper in range(16)]

        result = [self.enc_zero for _ in range(8)]
        for cond, out_bits in zip(upper_condition, selected_lower_outputs):
            result = [self.mux(cond, out_bit, res_bit) for out_bit, res_bit in zip(out_bits, result)]

        return result



def shift_rows(bits):
    assert len(bits) == 128
    rows = [[] for _ in range(4)]

    for col in range(4):
        for row in range(4):
            byte = bits[(4 * col + row) * 8 : (4 * col + row + 1) * 8]
            rows[row].append(byte)

    for row in range(4):
        rows[row] = rows[row][row:] + rows[row][:row]

    shifted = []
    for col in range(4):
        for row in range(4):
            shifted.extend(rows[row][col])
    return shifted


def inv_shift_rows(bits):
    assert len(bits) == 128
    rows = [[] for _ in range(4)]

    for col in range(4):
        for row in range(4):
            byte = bits[(4 * col + row) * 8 : (4 * col + row + 1) * 8]
            rows[row].append(byte)

    for row in range(4):
        rows[row] = rows[row][-row:] + rows[row][:-row]

    shifted = []
    for col in range(4):
        for row in range(4):
            shifted.extend(rows[row][col])
    return shifted


def xor_bits(engine, a, b):
    return [xor_bit(engine, x, y) for x, y in zip(a, b)]


def add_round_key(engine, state_bits, key_bits):
    return xor_bits(engine, state_bits, key_bits)


def xtime(engine, byte_bits, public_key, relinearization_key):
    shifted = byte_bits[1:] + [encrypt_bit(engine, public_key, 0)]
    msb = byte_bits[7]
    mask_bits = [(0x1B >> i) & 1 for i in range(8)]
    mask = [encrypt_bit(engine, public_key, b) for b in mask_bits]
    result = []
    for i in range(8):
        result.append(xor_bit(engine, shifted[i], and_bit(engine, msb, mask[i], relinearization_key)))
    return result


def mul_by_02(engine, byte_bits, public_key, relinearization_key):
    return xtime(engine, byte_bits, public_key, relinearization_key)


def mul_by_03(engine, byte_bits, public_key, relinearization_key):
    return xor_bits(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), byte_bits)


def mix_single_column(engine, column_bits, public_key, relinearization_key):
    b = column_bits
    r = [
        xor_bits(
            engine,
            xor_bits(engine, mul_by_02(engine, b[0], public_key, relinearization_key), mul_by_03(engine, b[1], public_key, relinearization_key)),
            xor_bits(engine, b[2], b[3]),
        ),
        xor_bits(
            engine,
            xor_bits(engine, b[0], mul_by_02(engine, b[1], public_key, relinearization_key)),
            xor_bits(engine, mul_by_03(engine, b[2], public_key, relinearization_key), b[3]),
        ),
        xor_bits(
            engine,
            xor_bits(engine, b[0], b[1]),
            xor_bits(engine, mul_by_02(engine, b[2], public_key, relinearization_key), mul_by_03(engine, b[3], public_key, relinearization_key)),
        ),
        xor_bits(
            engine,
            xor_bits(engine, mul_by_03(engine, b[0], public_key, relinearization_key), b[1]),
            xor_bits(engine, b[2], mul_by_02(engine, b[3], public_key, relinearization_key)),
        ),
    ]
    return r


def mix_columns(engine, state_bits, public_key, relinearization_key):
    new_state = []
    for col in range(4):
        column_bits = []
        for row in range(4):
            byte = state_bits[(4 * col + row) * 8 : (4 * col + row + 1) * 8]
            column_bits.append(byte)
        mixed = mix_single_column(engine, column_bits, public_key, relinearization_key)
        for byte in mixed:
            new_state.extend(byte)
    return new_state


def inv_mix_single_column(engine, column_bits, public_key, relinearization_key):
    b = column_bits

    def mul_by_09(engine, byte_bits, public_key, relinearization_key):
        return xor_bits(
            engine,
            mul_by_02(engine, mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key), public_key, relinearization_key),
            byte_bits,
        )

    def mul_by_0b(engine, byte_bits, public_key, relinearization_key):
        return xor_bits(
            engine,
            mul_by_02(engine, mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key), public_key, relinearization_key),
            xor_bits(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), byte_bits),
        )

    def mul_by_0d(engine, byte_bits, public_key, relinearization_key):
        return xor_bits(
            engine,
            mul_by_02(engine, mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key), public_key, relinearization_key),
            xor_bits(engine, mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key), byte_bits),
        )

    def mul_by_0e(engine, byte_bits, public_key, relinearization_key):
        return xor_bits(
            engine,
            mul_by_02(engine, mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key), public_key, relinearization_key),
            xor_bits(
                engine,
                mul_by_02(engine, mul_by_02(engine, byte_bits, public_key, relinearization_key), public_key, relinearization_key),
                mul_by_02(engine, byte_bits, public_key, relinearization_key),
            ),
        )

    r = [
        xor_bits(
            engine,
            xor_bits(
                engine,
                xor_bits(engine, mul_by_0e(engine, b[0], public_key, relinearization_key), mul_by_0b(engine, b[1], public_key, relinearization_key)),
                mul_by_0d(engine, b[2], public_key, relinearization_key),
            ),
            mul_by_09(engine, b[3], public_key, relinearization_key),
        ),
        xor_bits(
            engine,
            xor_bits(
                engine,
                xor_bits(engine, mul_by_09(engine, b[0], public_key, relinearization_key), mul_by_0e(engine, b[1], public_key, relinearization_key)),
                mul_by_0b(engine, b[2], public_key, relinearization_key),
            ),
            mul_by_0d(engine, b[3], public_key, relinearization_key),
        ),
        xor_bits(
            engine,
            xor_bits(
                engine,
                xor_bits(engine, mul_by_0d(engine, b[0], public_key, relinearization_key), mul_by_09(engine, b[1], public_key, relinearization_key)),
                mul_by_0e(engine, b[2], public_key, relinearization_key),
            ),
            mul_by_0b(engine, b[3], public_key, relinearization_key),
        ),
        xor_bits(
            engine,
            xor_bits(
                engine,
                xor_bits(engine, mul_by_0b(engine, b[0], public_key, relinearization_key), mul_by_0d(engine, b[1], public_key, relinearization_key)),
                mul_by_09(engine, b[2], public_key, relinearization_key),
            ),
            mul_by_0e(engine, b[3], public_key, relinearization_key),
        ),
    ]

    return r


def inv_mix_columns(engine, state_bits, public_key, relinearization_key):
    new_state = []
    for col in range(4):
        column_bits = []
        for row in range(4):
            byte = state_bits[(4 * col + row) * 8 : (4 * col + row + 1) * 8]
            column_bits.append(byte)
        mixed = inv_mix_single_column(engine, column_bits, public_key, relinearization_key)
        for byte in mixed:
            new_state.extend(byte)
    return new_state


def rot_word(word):
    return word[1:] + word[:1]


def sub_word(engine, word, sbox):
    new_word = []
    for i in range(4):
        byte_bits = word[i * 8 : (i + 1) * 8]
        subbed = sbox.lookup(byte_bits)
        new_word.extend(subbed)
    return new_word


def key_expansion(engine, enc_key_bits, sbox, public_key, relinearization_key):
    Rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
    Rcon_enc = []
    for r in Rcon:
        bits = [(r >> i) & 1 for i in range(8)]
        Rcon_enc.append([encrypt_bit(engine, public_key, b) for b in bits])

    words = []
    for i in range(4):
        words.append(enc_key_bits[i * 32 : (i + 1) * 32])

    for i in range(4, 44):
        temp = words[i - 1]
        if i % 4 == 0:
            temp = sub_word(engine, rot_word(temp), sbox)
            rcon_idx = (i // 4) - 1
            rcon_bits = Rcon_enc[rcon_idx] + [encrypt_bit(engine, public_key, 0)] * 24
            for j in range(32):
                temp[j] = xor_bit(engine, temp[j], rcon_bits[j])
        word = []
        for j in range(32):
            word.append(xor_bit(engine, words[i - 4][j], temp[j]))
        words.append(word)

    round_keys = []
    for r in range(11):
        rk = []
        for w in range(4):
            rk.extend(words[r * 4 + w])
        round_keys.append(rk)

    return round_keys


def decrypt_plaintext(engine, secret_key, enc_bits):
    decrypted = []
    for bit in enc_bits:
        decrypted.append(decrypt_bit(engine, secret_key, bit))
    return bits_to_bytes(decrypted)


def main():
    engine = Engine(use_bootstrap = True) # 혹은 비슷한 옵션

    secret_key = engine.create_secret_key()
    public_key = engine.create_public_key(secret_key)
    relinearization_key = engine.create_relinearization_key(secret_key)
    conjugation_key = engine.create_conjugation_key(secret_key)  # 추가
    bootstrap_key = engine.create_bootstrap_key(secret_key)      # 추가

    plaintext = b"ExampleAES128Txt"  #16바이트 평문
    key = b"ExampleAES128Key!"      #16바이트 키

    enc_key_bits = [encrypt_bit(engine, public_key, b) for b in bytes_to_bits(key)]

    sbox = SboxLUT(engine, public_key, relinearization_key, conjugation_key, bootstrap_key)
    inv_sbox = InvSboxLUT(engine, public_key, relinearization_key, conjugation_key, bootstrap_key)

    round_keys = key_expansion(engine, enc_key_bits, sbox, public_key, relinearization_key)

    enc_pt_bits = [encrypt_bit(engine, public_key, b) for b in bytes_to_bits(plaintext)]

    state = enc_pt_bits

    #AddRoundKey
    state = add_round_key(engine, state, round_keys[0])

    #1~9 라운드
    for rnd in range(1, 10):
        # SubBytes
        new_state = []
        for i in range(16):
            byte = state[i * 8 : (i + 1) * 8]
            subbed = sbox.lookup(byte)
            new_state.extend(subbed)
        state = new_state

        #ShiftRows
        state = shift_rows(state)

        #MixColumns
        state = mix_columns(engine, state, public_key, relinearization_key)

        #AddRoundKey
        state = add_round_key(engine, state, round_keys[rnd])

    #10라(MixColumns없음)
    new_state = []
    for i in range(16):
        byte = state[i * 8 : (i + 1) * 8]
        subbed = sbox.lookup(byte)
        new_state.extend(subbed)
    state = new_state

    state = shift_rows(state)
    state = add_round_key(engine, state, round_keys[10])

    #복호화 과정
    state = add_round_key(engine, state, round_keys[10])

    for rnd in reversed(range(1, 10)):
        state = inv_shift_rows(state)

        new_state = []
        for i in range(16):
            byte = state[i * 8 : (i + 1) * 8]
            subbed = inv_sbox.lookup(byte)
            new_state.extend(subbed)
        state = new_state

        state = add_round_key(engine, state, round_keys[rnd])

        state = inv_mix_columns(engine, state, public_key, relinearization_key)

    state = inv_shift_rows(state)

    new_state = []
    for i in range(16):
        byte = state[i * 8 : (i + 1) * 8]
        subbed = inv_sbox.lookup(byte)
        new_state.extend(subbed)
    state = new_state

    state = add_round_key(engine, state, round_keys[0])

    decrypted_bytes = decrypt_plaintext(engine, secret_key, state)

    print("복호화 결과 : ", decrypted_bytes)
    print("원본 평문  : ", plaintext)



if __name__ == "__main__":
    main()

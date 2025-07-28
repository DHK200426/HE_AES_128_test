import numpy as np
import random
from desilofhe import Engine

# ==============================================================================
# CELL 1: 사전 준비 및 LUT 생성
# ==============================================================================

n = 16
zeta = np.exp(-2 * np.pi * 1j / n)
jindices = np.arange(n)
iindices = np.arange(n)
exponents = np.outer(jindices, iindices)
code = zeta**jindices
u = zeta**exponents
u_conj_transpose = u.T.conj()
u_inv_manual = (1 / n) * u_conj_transpose

def k_axis_matrix_multiplication(C, U, k):
    alpha = C.ndim
    if k >= alpha:
        raise ValueError("Axis k is out of bounds for tensor C")
    c_indices = [chr(ord('a') + i) for i in range(alpha)]
    sum_index = c_indices[k]
    u_indices = sum_index + chr(ord('a') + alpha)
    d_indices = list(c_indices)
    d_indices[k] = u_indices[1]
    path = f"{''.join(c_indices)},{''.join(u_indices)}->{''.join(d_indices)}"
    return np.einsum(path, C, U)

def multiplekmul(C, U, k):
    if k == -1:
        return C
    new = k_axis_matrix_multiplication(C, U, k)
    return multiplekmul(new, U, k - 1)

def cforadd():
    indices = np.arange(16)
    D_numpy = np.bitwise_xor(indices[:, np.newaxis], indices)
    dd = zeta**D_numpy
    return multiplekmul(dd, u_inv_manual, 1)

s_box_values = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]
inv_s_box_values = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
]

s_box = np.array(s_box_values, dtype=np.uint8).reshape((16, 16))
inv_s_box = np.array(inv_s_box_values, dtype=np.uint8).reshape((16, 16))
sboxone = s_box >> 4
sboxtwo = s_box % 16
invsboxone = inv_s_box >> 4
invsboxtwo = inv_s_box % 16
cboxone = zeta**sboxone
cboxtwo = zeta**sboxtwo
inv_cboxone = zeta**invsboxone
inv_cboxtwo = zeta**invsboxtwo

def Cforsub():
    cone = multiplekmul(cboxone, u_inv_manual, 1)
    ctwo = multiplekmul(cboxtwo, u_inv_manual, 1)
    inv_cone = multiplekmul(inv_cboxone, u_inv_manual, 1)
    inv_ctwo = multiplekmul(inv_cboxtwo, u_inv_manual, 1)
    return cone, ctwo, inv_cone, inv_ctwo

def gf_multiply(a, b):
    p = 0
    irreducible_poly = 0x11B
    for _ in range(8):
        if b & 1:
            p ^= a
        high_bit_set = (a & 0x80)
        a <<= 1
        if high_bit_set:
            a ^= irreducible_poly
        b >>= 1
    return p & 0xFF

constants = [14, 11, 13, 9, 1, 3, 2]
inputs = list(range(256))
results_by_input = {i: {} for i in inputs}
results = []
for i in constants:
    temp = []
    for j in range(256):
        temp.append(gf_multiply(i, j))
    results.append(temp)
results = np.array(results)
results = results.reshape((7, 16, 16))
results1 = results >> 4
results2 = results % 16
results1 = zeta**results1
results2 = zeta**results2
cformult1 = []
cformult2 = []
for i in range(7):
    cone = multiplekmul(results1[i], u_inv_manual, 1)
    ctwo = multiplekmul(results2[i], u_inv_manual, 1)
    cformult1.append(cone)
    cformult2.append(ctwo)

Cforxor = cforadd()
Cforsboxup, Cforsboxdown, inv_Cforsboxup, inv_Cforsboxdown = Cforsub()

print("Setting up HE engine...")
engine = Engine(use_bootstrap=True, mode="gpu")


secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
rotation_key = engine.create_rotation_key(secret_key)
rotation_key_512 = [
    engine.create_fixed_rotation_key(secret_key, delta=512 * 0),
    engine.create_fixed_rotation_key(secret_key, delta=512 * 1),
    engine.create_fixed_rotation_key(secret_key, delta=512 * 2),
    engine.create_fixed_rotation_key(secret_key, delta=512 * 3)
]
conjugation_key = engine.create_conjugation_key(secret_key)
bootstrap_key = engine.create_bootstrap_key(secret_key, stage_count=3)
inv = [512 * 0, 512 * 5, -512 * 6, -512]
rotation_key_inv = [engine.create_fixed_rotation_key(secret_key, delta=i) for i in inv]
rotation_key_3072 = engine.create_fixed_rotation_key(secret_key, delta = 3072)
rotation_key_i2560 = engine.create_fixed_rotation_key(secret_key, delta = -2560)
rotation_key_i1024 = engine.create_fixed_rotation_key(secret_key, delta = -1024)
rotation_key_i512 = engine.create_fixed_rotation_key(secret_key, delta = -512)

def multiplyciphertexted(a,b,relinearization_key) :
    now = min(a.level, b.level)
    res = engine.multiply(a,b,relinearization_key)
    if now == 0 :
        res = engine.bootstrap(res, relinearization_key, conjugation_key, bootstrap_key)
    return res

def multiplywithplain(a,plain) :
    res = engine.multiply(a,plain)
    if a.level == 1 :
        res = engine.bootstrap(res, relinearization_key, conjugation_key, bootstrap_key)
    return res

print("HE setup complete.")

# ==============================================================================
# CELL 2: 키 생성 알고리즘
# ==============================================================================

RCON = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a
]

def key_expansion(key: bytes) -> list[list[int]]:
    if len(key) != 16:
        raise ValueError("Key must be 16 bytes (128 bits) long.")
    words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    for i in range(4, 44):
        temp = list(words[i - 1])
        if i % 4 == 0:
            temp.append(temp.pop(0))
            temp = [s_box_values[b] for b in temp]
            temp[0] ^= RCON[i // 4]
        new_word = [words[i - 4][b] ^ temp[b] for b in range(4)]
        words.append(new_word)
    return words


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


# ==============================================================================
# CELL 3: 동형 암호 연산 함수
# ==============================================================================

def calculate_powers_tree(ct_x, degree, engine, relin_key):
    if degree < 1:
        return {}
    powers = {1: ct_x}
    for i in range(1, degree):
        power_of_2 = 2**i
        if power_of_2 > degree:
            break
        prev_power = powers[power_of_2 // 2]
        powers[power_of_2] = engine.square(prev_power, relin_key)
    for i in range(3, degree + 1):
        if i in powers:
            continue
        binary_repr = bin(i)[2:]
        needed_powers = []
        for bit_idx, bit in enumerate(reversed(binary_repr)):
            if bit == '1':
                needed_powers.append(powers[2**bit_idx])
        current_res = needed_powers[0]
        for p_idx in range(1, len(needed_powers)):
            current_res = multiplyciphertexted(current_res, needed_powers[p_idx], relin_key)

        powers[i] = current_res
    return powers

def evaluate_univariate_polynomial(ct_x, coeffs, engine, relin_key):
    degree = len(coeffs) - 1
    powers_of_x = calculate_powers_tree(ct_x, degree, engine, relin_key)
    result = 0
    for i in range(1, degree + 1):
        if coeffs[i].any() != 0:
            term = multiplywithplain(powers_of_x[i], coeffs[i])
            result = engine.add(result, term)
    return result

def evaluate_16x16_lut(ct_a, ct_b, c, engine, relin_key):
    ct_Q_list = []
    for i in range(16):
        coeffs_for_Qi = c[i]
        ct_Qi = evaluate_univariate_polynomial(ct_b, coeffs_for_Qi, engine, relin_key)
        ct_Q_list.append(ct_Qi)
    powers_of_a = calculate_powers_tree(ct_a, 15, engine, relin_key)
    result = ct_Q_list[0]
    for i in range(1, 16):
        term = multiplyciphertexted(ct_Q_list[i], powers_of_a[i], relin_key)
        result = engine.add(result, term)
    return result

def evaluate_addroundkey(a, b, c, engine, relin_key):
    ct_Q_list = []
    for i in range(16):
        coeffs_for_Qi = c[i]
        key1 = np.array([1 for _ in range(8192)])
        res = np.array([complex(0) for _ in range(8192)])
        for k in range(16):
            temp = key1 * coeffs_for_Qi[k]
            key1 = key1 * b
            res += temp
        ct_Q_list.append(res)
    powers_of_a = calculate_powers_tree(a, 15, engine, relin_key)
    result = ct_Q_list[0]
    for i in range(1, 15):
        ct_Q_list_enc = engine.encrypt(ct_Q_list[i], public_key)
        term = multiplyciphertexted(ct_Q_list_enc, powers_of_a[i], relin_key)
        result = engine.add(result, term)
    return result

def LUTforaddroundkey(a, b):
    return evaluate_addroundkey(a, b, Cforxor, engine, relinearization_key)

def cforsm():
    Coeif = [[[[0 for _ in range(16)] for _ in range(16)] for _ in range(2)] for _ in range(4)]
    for i in range(4):
        for j in range(2):
            temp = cformult1 if j == 0 else cformult2
            for k in range(16):
                for s in range(16):
                    t = np.array([temp[6][i][j] for _ in range(512)] + [temp[5][i][j] for _ in range(512)] + [temp[4][i][j] for _ in range(1024)])
                    if i == 1:
                        t = np.roll(t, 512)
                    elif i == 2:
                        t = np.roll(t, 1024)
                    elif i == 3:
                        t = np.roll(t, -512)
                    t = t * 4
                    Coeif[i][j][k][s] = t
    return Coeif

def cforism():
    Coeif = [[[[0 for _ in range(16)] for _ in range(16)] for _ in range(2)] for _ in range(4)]
    for i in range(4):
        for j in range(2):
            temp = cformult1 if j == 0 else cformult2
            for k in range(16):
                for s in range(16):
                    t = np.array([temp[0][i][j] for _ in range(512)] + [temp[1][i][j] for _ in range(512)] + [temp[2][i][j] for _ in range(512)] + [temp[3][i][j] for _ in range(512)])
                    if i == 1:
                        t = np.roll(t, 512)
                    elif i == 2:
                        t = np.roll(t, 1024)
                    elif i == 3:
                        t = np.roll(t, -512)
                    t = t * 4
                    Coeif[i][j][k][s] = t
    return Coeif

cms = cforsm()
cps = cforism()

def shiftandaddandmasking(a):
    b = engine.rotate(a, rotation_key, 3072)
    c = evaluate_16x16_lut(a, b, Cforxor, engine, relinearization_key)
    d = engine.rotate(c, rotation_key, -2560)
    e = evaluate_16x16_lut(c, d, Cforxor, engine, relinearization_key)
    mask = np.zeros(8192)
    for i in [0, 2048, 4096, 6144]:
        mask[i:i + 512] = 1
    res = multiplywithplain(e, mask)
    return res

def mixcolumnshiftrow(a, b, cms_coeffs):
    x0up = evaluate_16x16_lut(a, b, cms_coeffs[0][0], engine, relinearization_key)
    x0down = evaluate_16x16_lut(a, b, cms_coeffs[0][1], engine, relinearization_key)
    x1up = evaluate_16x16_lut(a, b, cms_coeffs[1][0], engine, relinearization_key)
    x1down = evaluate_16x16_lut(a, b, cms_coeffs[1][1], engine, relinearization_key)
    x2up = evaluate_16x16_lut(a, b, cms_coeffs[2][0], engine, relinearization_key)
    x2down = evaluate_16x16_lut(a, b, cms_coeffs[2][1], engine, relinearization_key)
    x3up = evaluate_16x16_lut(a, b, cms_coeffs[3][0], engine, relinearization_key)
    x3down = evaluate_16x16_lut(a, b, cms_coeffs[3][1], engine, relinearization_key)
    up = [x0up, x1up, x2up, x3up]
    down = [x0down, x1down, x2down, x3down]
    upr = 0
    downr = 0
    for i in range(4):
        up[i] = shiftandaddandmasking(up[i])
        down[i] = shiftandaddandmasking(down[i])
        cc = engine.rotate(up[i], 512 * i, rotation_key_512[i])
        upr = engine.add(cc, upr)
        dd = engine.rotate(down[i], 512 * i, rotation_key_512[i])
        downr = engine.add(dd, downr)
    return upr, downr

def invshiftandaddandmasking(a):
    b = engine.rotate(a, rotation_key, -1024)
    c = evaluate_16x16_lut(a, b, Cforxor, engine, relinearization_key)
    d = engine.rotate(c, rotation_key, -512)
    e = evaluate_16x16_lut(c, d, Cforxor, engine, relinearization_key)
    mask = np.zeros(8192)
    for i in [0, 2048, 4096, 6144]:
        mask[i:i + 512] = 1
    res = multiplywithplain(e, mask)
    return res

def invmixcolumnshiftrow(a, b, cms_coeffs):
    x0up = evaluate_16x16_lut(a, b, cms_coeffs[0][0], engine, relinearization_key)
    x0down = evaluate_16x16_lut(a, b, cms_coeffs[0][1], engine, relinearization_key)
    x1up = evaluate_16x16_lut(a, b, cms_coeffs[1][0], engine, relinearization_key)
    x1down = evaluate_16x16_lut(a, b, cms_coeffs[1][1], engine, relinearization_key)
    x2up = evaluate_16x16_lut(a, b, cms_coeffs[2][0], engine, relinearization_key)
    x2down = evaluate_16x16_lut(a, b, cms_coeffs[2][1], engine, relinearization_key)
    x3up = evaluate_16x16_lut(a, b, cms_coeffs[3][0], engine, relinearization_key)
    x3down = evaluate_16x16_lut(a, b, cms_coeffs[3][1], engine, relinearization_key)
    up = [x0up, x1up, x2up, x3up]
    down = [x0down, x1down, x2down, x3down]
    upr = 0
    downr = 0
    for i in range(4):
        up[i] = invshiftandaddandmasking(up[i])
        down[i] = invshiftandaddandmasking(down[i])
        cc = engine.rotate(up[i], inv[i], rotation_key_inv[i])
        upr = engine.add(cc, upr)
        dd = engine.rotate(down[i], inv[i], rotation_key_inv[i])
        downr = engine.add(dd, downr)
    return upr, downr

def LUTforsbox(a, b, check):
    c1 = Cforsboxup if check else inv_Cforsboxup
    c2 = Cforsboxdown if check else inv_Cforsboxdown
    r1 = evaluate_16x16_lut(a, b, c1, engine, relinearization_key)
    r2 = evaluate_16x16_lut(a, b, c2, engine, relinearization_key)
    return r1, r2


if __name__ == "__main__" :
    #512개의 평문을 8192개의 슬롯에 a00b00...a10b10...a20b20식으로 열우선으로 배치
    plaintext = [[random.randrange(256) for _ in range(16)] for _ in range(512)] 
    data = [] #슬롯
    
    for i in range(16) :
        for j in range(512) : 
            data.append(plaintext[j][i])
    data = np.array(data)
    dataup = data >> 4
    datadown = data % 16
    dataup = zeta ** dataup
    datadown = zeta ** datadown ##복소평면으로 매핑
    cryptedup = engine.encrypt(dataup, public_key)
    crypteddown = engine.encrypt(datadown, public_key)
    """--------------------------------firstround-----------------------------------"""
    addup = LUTforaddroundkey(cryptedup,roundkeysone[0])
    adddown = LUTforaddroundkey(crypteddown,roundkeystwo[0])
    """--------------------------------n-1 round------------------------------------"""
    addup = LUTforaddroundkey(addup, roundkeysone[0])
    adddown = LUTforaddroundkey(adddown, roundkeystwo[0])
    "-------------------------------------------------------"
    check1 = engine.decrypt(cryptedup,secret_key)
    check2 = engine.decrypt(crypteddown, secret_key)
    print(check1)
    print(check2)
    res1 = engine.decrypt(addup, secret_key)
    res2 = engine.decrypt(adddown, secret_key)
    print(res1)
    print(res2)

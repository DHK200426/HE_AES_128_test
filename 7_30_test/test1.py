"""사전준비 LUT만들기 """

from desilofhe import Engine
import numpy as np
import random
n = 16

zeta = np.exp(-2*np.pi *1j /n)

jindices = np.arange(n)
iindices = np.arange(n)
exponents = np.outer(jindices,iindices)
code = zeta ** jindices
u = zeta ** exponents

u_conj_transpose = u.T.conj()
u_inv_manual = (1/n) * u_conj_transpose

    

def k_axis_matrix_multiplication(C, U, k) :
    """
    다차원 행렬 C의 k번째 축에 대해 행렬 U를 곱하는 ⊠ₖ 연산을 구현합니다.
    D = C ⊠ₖ U
    """
    
    # C의 차원 수 확인
    alpha = C.ndim
    if k >= alpha:
        raise ValueError("Axis k is out of bounds for tensor C")
        
    # einsum 경로 문자열을 동적으로 생성
    # 예: C가 3차원(ijk), U가 2차원(jl), k=1 이면 'ijk,jl->ilk'
    
    # 1. 입력 인덱스 문자열 생성
    c_indices = [chr(ord('a') + i) for i in range(alpha)]
    sum_index = c_indices[k]
    u_indices = sum_index + chr(ord('a') + alpha) # 합쳐질 인덱스와 새로운 출력 인덱스
    
    # 2. 출력 인덱스 문자열 생성
    d_indices = list(c_indices)
    d_indices[k] = u_indices[1] # k번째 축을 U의 두 번째 인덱스로 교체
    
    # 3. einsum 경로 완성
    path = f"{''.join(c_indices)},{''.join(u_indices)}->{''.join(d_indices)}"
    
    # einsum을 이용해 텐서 곱셈 수행
    return np.einsum(path, C, U)

def multiplekmul(C,U,k) :
    if k == -1 :
        return C
    new = k_axis_matrix_multiplication(C, U, k)
    return multiplekmul(new, U, k-1)

"""-----------------------------------------------------------------------------------------------------------------"""
def cforadd() : ## input : 4 , 4bits -> output : 4 bits 
    indices = np.arange(16)
    D_numpy = np.bitwise_xor(indices[:,np.newaxis], indices)
    dd = zeta ** D_numpy
    return multiplekmul(dd, u_inv_manual,1)
"""--------------------------------------------------------------------------------------------------------------------"""
#LUT for subbytes (4,4) -> (4,4)
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

s_box = np.array(s_box_values, dtype = np.uint8).reshape((16,16))
inv_s_box = np.array(inv_s_box_values,dtype = np.uint8).reshape((16,16))
sboxone = s_box >> 4 ##각 원소 상위 4비트
sboxtwo = s_box % 16 ##각 원소 하의 4비트
invsboxone = inv_s_box >> 4
invsboxtwo = inv_s_box % 16

cboxone = zeta ** sboxone ##실수를 복소평면 단위원 위로
cboxtwo = zeta ** sboxtwo ##실수를 복소평면 단위원 위로 
inv_cboxone = zeta ** invsboxone
inv_cboxtwo = zeta ** invsboxtwo

def Cforsub() : ## 8 -> 8 return two coefficient matrix 
    cone = multiplekmul(cboxone, u_inv_manual,1)
    ctwo = multiplekmul(cboxtwo, u_inv_manual,1)
    inv_cone = multiplekmul(inv_cboxone,u_inv_manual,1)
    inv_ctwo = multiplekmul(inv_cboxtwo, u_inv_manual,1)
    return cone, ctwo,inv_cone,inv_ctwo

def gf_multiply(a, b):
    """AES의 GF(2^8)상에서 두 숫자를 곱하는 함수"""
    p = 0
    # 0x11B는 AES의 기약 다항식 x^8 + x^4 + x^3 + x + 1에 해당
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

# 요청된 상수들
constants = [14, 11, 13, 9, 1, 3, 2]
# 0부터 255까지의 모든 입력값
inputs = list(range(256))

# 입력값을 키(key)로 하는 딕셔너리 생성
results_by_input = {i: {} for i in inputs}

results = []
for i in constants :
    temp = []
    for j in range(256) :
        temp.append(gf_multiply(i,j))
    results.append(temp)
results = np.array(results)
results = results.reshape((7,16,16))
results1 = results >> 4
results2 = results % 16
results1 = zeta ** results1
results2 = zeta ** results2

cformult1 = []
cformult2 = []

for i in range(7) :
    cone = multiplekmul(results1[i], u_inv_manual,1)
    ctwo = multiplekmul(results2[i], u_inv_manual,1)
    cformult1.append(cone)
    cformult2.append(ctwo)

Cforxor = cforadd()
Cforsboxup, Cforsboxdown,inv_Cforsboxup, inv_Cforsboxdown = Cforsub() 


"""cfromult1, cformult2, cforxor, cforsboxup, cforsboxdown,inv_cforsboxup, inv_cforsboxdown"""

"""cfor mult1 ,2 -> (7,16,16) 각각 원소 14 11 13 9 1 3 2와 대응됨, 상위 4비트 하위 4비트 입력받아서 각각 gf(2^8)에서의 곱셈 결과를 리턴하는 계수
    cforxor (4,4) - > 4를 하는 단순 4비트 4비트 xor결과를 리턴
    cforsboxup,down -> 4비트 4비트를 입력받아해당 자리에 있는 upper 4 bit lower 4bit를 리턴하는 LUT
"""

engine = Engine(use_bootstrap = True)
secret_key = engine.create_secret_key()
public_key = engine.create_public_key(secret_key)
relinearization_key = engine.create_relinearization_key(secret_key)
bootstrap_key = engine.create_bootstrap_key(secret_key)
conjugation_key = engine.create_conjugation_key(secret_key)
rotation_key = engine.create_rotation_key(secret_key)
rotation_key_512 = [engine.create_fixed_rotation_key(secret_key, delta=512*0),
engine.create_fixed_rotation_key(secret_key, delta=512*1),
engine.create_fixed_rotation_key(secret_key, delta=512*2),
engine.create_fixed_rotation_key(secret_key, delta=512*3)
]

inv = [512*0, 512*5,-512*6,-512]
rotation_key_inv = [engine.create_fixed_rotation_key(secret_key, delta = i) for i in inv]
rotation_key_3072 = engine.create_fixed_rotation_key(secret_key, delta = 3072)
rotation_key_i2560 = engine.create_fixed_rotation_key(secret_key, delta = -2560)
rotation_key_i1024 = engine.create_fixed_rotation_key(secret_key, delta = -1024)
rotation_key_i512 = engine.create_fixed_rotation_key(secret_key, delta = -512)
count = 0

rotation_key_just = [engine.create_fixed_rotation_key(secret_key, delta = -2048),engine.create_fixed_rotation_key(secret_key, delta = -4096),engine.create_fixed_rotation_key(secret_key, delta = 2048)]
def multiplyciphertexted(a,b,relinearization_key) :
    now = min(a.level, b.level)
    res = engine.multiply(a,b,relinearization_key)
    if now == 1 :
        count += 1
        res = engine.bootstrap(res, relinearization_key, conjugation_key, bootstrap_key)
    return res

def multiplywithplain(a,plain) :
    res = engine.multiply(a,plain)
    if a.level == 1 :
        count += 1
        res = engine.bootstrap(res, relinearization_key, conjugation_key, bootstrap_key)
    return res

"""----------------------------------------------------------------------------------------------------------------------------"""

#키생성 알고리즘
# Round Constants: 라운드 상수로, g 함수에서 사용됨
RCON = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a
]

def key_expansion(key: bytes) -> list[list[int]]:
    """AES-128 키를 받아 라운드 키 스케줄을 생성합니다."""
    if len(key) != 16:
        raise ValueError("Key must be 16 bytes (128 bits) long.")

    # 마스터 키를 4개의 32비트 워드로 나눔
    words = [list(key[i:i+4]) for i in range(0, 16, 4)]
    
    # 총 44개의 워드를 생성하기 위해 루프 실행 (11 라운드 * 4 워드/라운드)
    for i in range(4, 44):
        temp = list(words[i-1]) # 이전 워드를 복사

        # i가 4의 배수일 때 g 함수 적용
        if i % 4 == 0:
            # 1. RotWord: 1바이트씩 왼쪽 순환 이동
            temp.append(temp.pop(0))

            # 2. SubWord: S-Box를 이용한 바이트 치환
            temp = [s_box_values[b] for b in temp]

            # 3. Rcon XOR: 라운드 상수와 XOR
            temp[0] ^= RCON[i // 4]
        
        # 새로운 워드 계산: W[i] = W[i-4] XOR temp
        new_word = [words[i-4][b] ^ temp[b] for b in range(4)]
        words.append(new_word)
        
    return words

# --- 예제 실행 ---
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
        rkone = [rkone[i] for i in range(16) for _ in range(512)]
        rktwo = [rktwo[i] for i in range(16) for _ in range(512)]
        roundkeysone.append(rkone)
        roundkeystwo.append(rktwo)


"""---------------------------------------------------------------------------------------------------------------------------"""

# --- 가정: desilofhe 라이브러리와 키가 이미 설정되어 있다고 가정 ---
# class Engine:
#     def multiply(self, ct1, ct2, relin_key): ...
#     def square(self, ct, relin_key): ...
# engine = Engine()
# relin_key = ...
# ---------------------------------------------------------------
def calculate_powers_tree(ct_x, degree, engine, relin_key):
    """
    트리 구조를 이용해 x^1, x^2, ..., x^degree를 최소한의 곱셈 깊이로 계산합니다.
    """
    if degree < 1:
        return {}

    powers = {1: ct_x}

    # 1. 먼저 2의 거듭제곱 항들(x^2, x^4, x^8, ...)을 계산합니다.
    # 이것이 트리의 기본 골격이 되며, 곱셈 깊이를 log(N)으로 줄여줍니다.
    for i in range(1, degree):
        power_of_2 = 2**i
        if power_of_2 > degree:
            break
        # x^(2^i) = x^(2^(i-1)) * x^(2^(i-1))
        prev_power = powers[power_of_2 // 2]
        powers[power_of_2] = engine.square(prev_power, relin_key)

    # 2. 계산된 항들을 조합하여 나머지 모든 항을 계산합니다.
    # 예: x^11 = x^8 * x^2 * x^1
    for i in range(3, degree + 1):
        if i in powers:
            continue # 이미 계산된 항은 건너뛰기
        
        # i를 이진수로 분해하여 필요한 항들을 찾음
        binary_repr = bin(i)[2:]
        
        needed_powers = []
        for bit_idx, bit in enumerate(reversed(binary_repr)):
            if bit == '1':
                needed_powers.append(powers[2**bit_idx])
        
        # 필요한 항들을 곱하여 현재 항을 계산
        current_res = needed_powers[0]
        for p_idx in range(1, len(needed_powers)):
            current_res = multiplyciphertexted(current_res, needed_powers[p_idx],relin_key)
        powers[i] = current_res
            
    return powers

def evaluate_univariate_polynomial(ct_x, coeffs, engine, relin_key):
    """
    BSGS로 계산된 거듭제곱을 사용해 단일 변수 다항식을 효율적으로 평가합니다.
    P(x) = c_0 + c_1*x + c_2*x^2 + ...
    """
    degree = len(coeffs) - 1
    if type(coeffs[0]) != np.ndarray : 
        passing = True 
    else :
        passing = False
    
    # x의 모든 거듭제곱을 효율적으로 미리 계산
    powers_of_x = calculate_powers_tree(ct_x, degree, engine, relin_key)
    
    # c_0 항으로 결과 초기화
    if passing :
        result = coeffs[0]
    else :
        result = coeffs[0]
        result = engine.encode(result)
    # c_1*x + c_2*x^2 + ... 항들을 더해줌
    for i in range(1, degree + 1):
        if passing :
            cc = np.array([coeffs[i] for _ in range(8192)])
        else :
            cc = coeffs[i]
        cc = engine.encode(cc)
        term = multiplywithplain(powers_of_x[i], cc)
        result = engine.add(result, term)
            
    return result


def evaluate_16x16_lut(ct_a, ct_b, c, engine, relin_key):
    """
    (메인 함수)
    두 니블 입력(ct_a, ct_b)과 16x16 계수(c)로 다변수 다항식을 평가합니다.
    """
    # P(a, b) = Q_0(b) + Q_1(b)*a + Q_2(b)*a^2 + ...
    # 이 구조를 계산합니다.

    # 1. 먼저 b에 대한 다항식들(Q_0, ..., Q_15)을 각각 평가합니다.
    # 결과는 암호문 리스트 [ct_Q0, ct_Q1, ..., ct_Q15]가 됩니다.
    ct_Q_list = []
    for i in range(16):
        # c의 i번째 행이 Qi(b)의 계수가 됨
        coeffs_for_Qi = np.array(c[i])
        ct_Qi = evaluate_univariate_polynomial(ct_b, coeffs_for_Qi, engine, relin_key)
        ct_Q_list.append(ct_Qi)
        
    # 2. a의 거듭제곱(a^1, ..., a^15)을 효율적으로 미리 계산합니다.
    powers_of_a = calculate_powers_tree(ct_a, 15, engine, relin_key)
    
    # 3. 암호문 계수(ct_Q_list)와 a의 거듭제곱을 곱하여 최종 합계를 구합니다.
    # result = ct_Q0 + ct_Q1*a^1 + ct_Q2*a^2 + ...
    result = ct_Q_list[0] # Q_0(b) * a^0 항

    for i in range(1, 16):
        # ct_Qi * a^i 계산 (암호문-암호문 곱셈, 비용이 매우 높음)
        term = multiplyciphertexted(ct_Q_list[i], powers_of_a[i], relin_key)
        result = engine.add(result, term)
        
    return result
def evaluate_addroundkey(ct, key, c, engine, relin_key) :
    ct_Q_list = []
    for i in range(16) :
        coeffs_for_Qi = c[i]
        key1 = np.array([1 for _ in range(8192)])
        res = np.array([0 for _ in range(8192)])
        for k in range(16) :
            temp = key1 * coeffs_for_Qi[k]
            key1 = key1 * key
            res = res + temp
        res = engine.encrypt(res,public_key)
        ct_Q_list.append(res)
    powers_of_a = calculate_powers_tree(ct, 15, engine, relin_key)
    result = ct_Q_list[0]
    for i in range(1,16) :
        term = multiplyciphertexted(powers_of_a[i], ct_Q_list[i],relin_key)
        result = engine.add(result, term)
    return result
    

def LUTforaddroundkey(a,b) :
    return evaluate_addroundkey(a,b,Cforxor,engine, relinearization_key)

def cforsm() :
    Coeif = [[[[0 for _ in range(16)] for _ in range(16)] for _ in range(2)] for _ in range(4)]
    for i in range(4) :
        for j in range(2) :
            if j == 0 :
                temp = cformult1
            else :
                temp = cformult2
            for k in range(16) :
                for s in range(16) :
                    t = np.array([temp[6][k][s] for _ in range(512)] + [temp[5][k][s] for _ in range(512)] + [temp[4][k][s] for _ in range(1024)])

                    if i == 1 :
                        t = np.roll(t, 512)
                    elif i == 2 :
                        t = np.roll(t,1024)
                    elif i == 3 :
                        t = np.roll(t,-512)
                    z = []
                    for _ in range(4) :
                        z = np.append(z,t)
                    Coeif[i][j][k][s] = z
    return Coeif

def cforism() :
    Coeif = [[[[0 for _ in range(16)] for _ in range(16)] for _ in range(2)] for _ in range(4)]
    for i in range(4) :
        for j in range(2) :
            if j == 0 :
                temp = cformult1
            else :
                temp = cformult2
            for k in range(16) :
                for s in range(16) :
                    t = np.array([temp[0][k][s] for _ in range(512)] + [temp[1][k][s] for _ in range(512)] + [temp[2][k][s] for _ in range(512)] + [temp[3][k][s] for _ in range(512)])

                    if i == 1 :
                        t = np.roll(t, 512)
                    elif i == 2 :
                        t = np.roll(t,1024)
                    elif i == 3 :
                        t = np.roll(t,-512)

                    z = []
                    for _ in range(4) :
                        z = np.append(z,t)
                    Coeif[i][j][k][s] = z
    return Coeif
    
cms = cforsm()
cps = cforism()
def shiftandaddandmasking(a) :
    b = engine.rotate(a,rotation_key_3072)
    c = evaluate_16x16_lut(a,b,Cforxor,engine,relinearization_key)
    d = engine.rotate(c,rotation_key_i2560)
    e = evaluate_16x16_lut(c,d,Cforxor,engine, relinearization_key)
    mask = np.array([0 for _ in range(8192)])
    for i in [0,2048,4096,6144] :
        mask[i : i+512] = 1
    res = multiplywithplain(e,mask)
    return res
def mixcolumnshiftrow(a,b,cms) :
    x0up = evaluate_16x16_lut(a,b,cms[0][0],engine,relinearization_key)
    x0down = evaluate_16x16_lut(a,b,cms[0][1],engine,relinearization_key)
    x1up = evaluate_16x16_lut(a,b,cms[1][0],engine,relinearization_key)
    x1down = evaluate_16x16_lut(a,b,cms[1][1],engine,relinearization_key)    
    x2up = evaluate_16x16_lut(a,b,cms[2][0],engine,relinearization_key)
    x2down = evaluate_16x16_lut(a,b,cms[2][1],engine,relinearization_key)    
    x3up = evaluate_16x16_lut(a,b,cms[3][0],engine,relinearization_key)
    x3down = evaluate_16x16_lut(a,b,cms[3][1],engine,relinearization_key)
    up = [x0up,x1up,x2up,x3up]
    down = [x0down,x1down,x2down,x3down]
    upr = 0
    downr = 0
    for i in range(4) :
        up[i] = shiftandaddandmasking(up[i])
        down[i] = shiftandaddandmasking(down[i])
        cc = engine.rotate(up[i],rotation_key_512[i])
        upr = engine.add(cc,upr)
        dd = engine.rotate(down[i],rotation_key_512[i])
        downr = engine.add(dd,downr)
    return upr, downr
def invshiftandaddandmasking(a) :
    b = engine.rotate(a,rotation_key_i1024)
    c = evaluate_16x16_lut(a,b,Cforxor,engine, relinearization_key)
    d = engine.rotate(c,rotation_key_i512)
    e = evaluate_16x16_lut(c,d,Cforxor,engine, relinearization_key)
    mask = [0 for _ in range(8192)]
    for i in [0,2048,4096,6144] :
        mask[i : i+512] = 1
    res = multiplywithplain(e,mask)
    return res
def invmixcolumnshiftrow(a,b,cms) :#Cps 
    x0up = evaluate_16x16_lut(a,b,cms[0][0],engine,relinearization_key)
    x0down = evaluate_16x16_lut(a,b,cms[0][1],engine,relinearization_key)
    x1up = evaluate_16x16_lut(a,b,cms[1][0],engine,relinearization_key)
    x1down = evaluate_16x16_lut(a,b,cms[1][1],engine,relinearization_key)    
    x2up = evaluate_16x16_lut(a,b,cms[2][0],engine,relinearization_key)
    x2down = evaluate_16x16_lut(a,b,cms[2][1],engine,relinearization_key)    
    x3up = evaluate_16x16_lut(a,b,cms[3][0],engine,relinearization_key)
    x3down = evaluate_16x16_lut(a,b,cms[3][1],engine,relinearization_key)
    up = [x0up,x1up,x2up,x3up]
    down = [x0down,x1down,x2down,x3down]
    upr = 0
    downr = 0
    for i in range(4) :
        up[i] = invshiftandaddandmasking(up[i])
        down[i] = invshiftandaddandmasking(down[i])
        cc = engine.rotate(up[i],rotation_key_inv[i])
        upr = engine.add(cc,upr)
        dd = engine.rotate(down[i],rotation_key_inv[i])
        downr = engine.add(dd,downr)
    return upr, downr

def LUTforsbox(a,b,check) :
    if check == True :
        c1 = Cforsboxup
        c2 = Cforsboxdown
    else :
        c1 = inv_Cforsboxup
        c2 = inv_Cforsboxdown
    r1 = evaluate_16x16_lut(a,b,c1,engine, relinearization_key)
    r2 = evaluate_16x16_lut(a,b,c2,engine, relinearization_key)

    return r1, r2
def justshift(a) :
    mask1 = np.array([0 for _ in range(8192)])
    for i in [0,2048,4096,6144] :
        mask1[i:i+512] = 1
    mask2 = np.roll(mask1,512)
    mask3 = np.roll(mask2,512)
    mask4 = np.roll(mask3,512)

    new1 = multiplywithplain(a,mask1)
    new2 = multiplywithplain(a,mask2)
    new3 = multiplywithplain(a,mask3)
    new4 = multiplywithplain(a,mask4)

    new2 = engine.rotate(new2, rotation_key_just[0])
    new3 = engine.rotate(new3, rotation_key_just[1])
    new4 = engine.rotate(new4, rotation_key_just[2])

    new1 = engine.add(new1,new2)
    new1 = engine.add(new1,new3)
    new1 = engine.add(new1,new4)
    return new1

def invjustshift(a) :
    mask1 = np.array([0 for _ in range(8192)])
    for i in [0,2048,4096,6144] :
        mask1[i:i+512] = 1
    mask2 = np.roll(mask1,512)
    mask3 = np.roll(mask2,512)
    mask4 = np.roll(mask3,512)

    new1 = multiplywithplain(a,mask1)
    new2 = multiplywithplain(a,mask2)
    new3 = multiplywithplain(a,mask3)
    new4 = multiplywithplain(a,mask4)

    new2 = engine.rotate(new2, rotation_key_just[2])
    new3 = engine.rotate(new3, rotation_key_just[1])
    new4 = engine.rotate(new4, rotation_key_just[0])

    new1 = engine.add(new1,new2)
    new1 = engine.add(new1,new3)
    new1 = engine.add(new1,new4)
    return new1
"""------------------------------------------------------------------------------------------------------------------------------"""
def check(a,b) :
    res = 0
    for i in range(16) :
        for j in range(16) :
            res += Cforxor[i][j]*(a**i) * (b**j)
    return res

def decode(a) :
    cha = 100
    rr = -1
    for i in range(16) :
        res = abs(a - code[i])
        if cha > res :
            rr = i
            cha = res

    return rr

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
    
    sbup, sbdown = LUTforsbox(cryptedup, crypteddown,True)
    resup, resdown = LUTforsbox(sbup, sbdown, False)
    r1 = engine.decrypt(resup, secret_key)
    r2 = engine.decrypt(resdown, secret_key)
    print(r1[:4095])
    print(dataup_encoded[:4095])
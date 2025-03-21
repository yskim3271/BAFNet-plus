import torch
import numpy as np
import matplotlib.pyplot as plt
from models.seconformer_orig import seconformer as seconformer_orig
from models.seconformer import seconformer as seconformer_new
import time

def test_seconformer_implementations():
    """
    SEConformer 원본 구현과 새 구현의 차이점을 테스트합니다.
    동일한 입력에 대해 두 모델의 출력을 비교합니다.
    """
    # 테스트 파라미터 설정
    sample_rate = 16000
    batch_size = 2
    channels = 1
    length = 16000  # 1초 오디오
    
    # 동일한 랜덤 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 랜덤 입력 생성 (오디오 신호 시뮬레이션)
    input_signal = torch.randn(batch_size, channels, length)
    
    # 두 모델 초기화 (동일한 파라미터로)
    common_params = {
        'chin': 1,  # 입력 채널
        'chout': 1,  # 출력 채널
        'hidden': 32,  # 히든 차원
        'depth': 4,  # 인코더/디코더 깊이
        'conformer_dim': 256,  # Conformer 차원
        'conformer_ffn_dim': 256,  # Conformer FFN 차원
        'conformer_num_attention_heads': 4,  # Conformer 어텐션 헤드 수
        'conformer_depth': 1,  # Conformer 깊이
        'depthwise_conv_kernel_size': 31,  # 컨볼루션 커널 크기
        'kernel_size': 8,  # 인코더/디코더 커널 크기
        'stride': 4,  # 인코더/디코더 스트라이드
        'resample': 4,  # 리샘플링 비율
        'growth': 2,  # 성장률
        'dropout': 0.1,  # 드롭아웃 비율
        'rescale': 0.1,  # 리스케일 값
        'normalize': True,  # 정규화 여부
        'sample_rate': sample_rate,  # 샘플링 레이트
    }
    
    # 원본 모델
    model_orig = seconformer_orig(**common_params)
    
    # # 새 모델 (causal=False로 비인과적 설정)
    # model_new = seconformer_new(**common_params, causal=False)
    
    # # 새 모델 (causal=True로 인과적 설정)
    # model_new_causal = seconformer_new(**common_params, causal=True)
    
    # 평가 모드로 설정
    model_orig.eval()
    # model_new.eval()
    # model_new_causal.eval()
    
    # 추론 시간 및 결과 비교
    with torch.no_grad():
        # 원본 모델
        # start_time = time.time()
        output_orig = model_orig(input_signal)
        # orig_time = time.time() - start_time
        
        # 새 모델 (비인과적)
        # start_time = time.time()
        # output_new = model_new(input_signal)
        # new_time = time.time() - start_time
        
        # 새 모델 (인과적)
        # start_time = time.time()
        # output_new_causal = model_new_causal(input_signal)
        # new_causal_time = time.time() - start_time
    
    # 결과 출력
    print(f"원본 모델 출력 크기: {output_orig.shape}")
    # print(f"새 모델 출력 크기 (비인과적): {output_new.shape}")
    # print(f"새 모델 출력 크기 (인과적): {output_new_causal.shape}")
    
    # print(f"\n원본 모델 처리 시간: {orig_time:.4f}초")
    # print(f"새 모델 처리 시간 (비인과적): {new_time:.4f}초")
    # print(f"새 모델 처리 시간 (인과적): {new_causal_time:.4f}초")
    
    # MSE 계산
    # mse_orig_vs_new = torch.mean((output_orig - output_new) ** 2)
    # mse_orig_vs_causal = torch.mean((output_orig - output_new_causal) ** 2)
    # mse_new_vs_causal = torch.mean((output_new - output_new_causal) ** 2)
    
    # print(f"\nMSE (원본 vs 새 비인과): {mse_orig_vs_new:.6f}")
    # print(f"MSE (원본 vs 새 인과): {mse_orig_vs_causal:.6f}")
    # print(f"MSE (새 비인과 vs 새 인과): {mse_new_vs_causal:.6f}")

if __name__ == "__main__":
    test_seconformer_implementations() 
# BAFNet Alpha Fusion의 수학적 해석

## 목적

이 문서는 현재 `BAFNet`이 `alpha`를 계산하는 방식의 수학적 의미를 정리하고, 이를 보다 대칭적이고 해석 가능한 fusion 구조로 확장할 때 어떤 수학적 해석이 가능한지 설명한다. 특히 `bcs`와 `acs`의 상대 reliability, branch 간 scale mismatch, 그리고 causal calibration의 역할을 함께 다룬다.

핵심 질문은 다음과 같다.

- 현재의 `alpha`는 무엇을 의미하는가?
- `acs` 정보만으로 gate를 계산하는 것이 왜 비대칭적인가?
- `bcs` 정보와 causal calibration을 함께 도입하면 어떤 수학적 의미가 추가되는가?

## 현재 BAFNet의 결합 방식

현재 `BAFNet`은 두 backbone 출력의 complex spectrogram을 정규화한 뒤 `alpha`로 선형 결합한다.

$$
\tilde{S}^{bcs}_{f,t} = \text{normalize}(S^{bcs}_{f,t}), \qquad
\tilde{S}^{acs}_{f,t} = \text{normalize}(S^{acs}_{f,t})
$$

$$
\hat{S}_{f,t}
=
\alpha_{f,t}\,\tilde{S}^{bcs}_{f,t}
+
(1-\alpha_{f,t})\,\tilde{S}^{acs}_{f,t}
$$

그 후 평균 에너지로 다시 scale을 복원한다.

$$
S^{out}_{f,t} = E_{avg}\,\hat{S}_{f,t}
$$

여기서 현재 `alpha`는 `acs` branch가 만든 magnitude mask로부터만 계산된다.

$$
\alpha_{f,t} = g\!\left(m^{acs}_{f,t}\right)
$$

여기서

- $m^{acs}_{f,t}$: `acs` backbone이 만든 mask 또는 magnitude ratio 기반 신호
- $g(\cdot)$: 작은 CNN + sigmoid 계열의 gate 함수

즉, 현재 구조에서 `alpha`는 두 branch의 상대 reliability를 직접 나타내는 weight라기보다, `acs`가 만든 feature에만 condition된 비대칭 gate이다.

## 현재 구조의 해석

현재 구조는 다음과 같이 해석할 수 있다.

1. `acs` branch가 masking 결과로부터 fusion에 사용할 feature를 만든다.
2. 그 feature만을 바탕으로 `bcs` 결과와 `acs` 결과를 섞는다.

이 구조는 실용적으로는 동작할 수 있지만, 수학적으로는 약간 비대칭적이다. 최종 출력은 두 branch의 결합인데, 결합 비율을 정하는 정보는 사실상 한 branch에서만 온다.

즉 지금의 식은

$$
\hat{S}_{f,t}
=
g(m^{acs}_{f,t})\,\tilde{S}^{bcs}_{f,t}
+
\left(1-g(m^{acs}_{f,t})\right)\,\tilde{S}^{acs}_{f,t}
$$

형태이며, 핵심적인 점은 gate의 근거가 `acs` 쪽 정보에만 있다는 것이다. 따라서 이 식은 두 추정치의 상대적 우열을 대칭적으로 비교한 결과라기보다, `acs`-conditioned gate로 두 출력을 혼합하는 구조에 가깝다.

## 개선 모델의 수학적 해석

개선 모델을 가장 자연스럽게 해석하는 방법은, `bcs`와 `acs`를 동일한 clean target $S_{f,t}$에 대한 두 개의 estimator로 보는 것이다.

$$
S^{bcs}_{f,t} = S_{f,t} + \epsilon_b
$$

$$
S^{acs}_{f,t} = S_{f,t} + \epsilon_a
$$

여기서 $\epsilon_b, \epsilon_a$는 각 branch의 오차이다.

이때 개선 모델은 다음 두 단계를 갖는 구조로 해석할 수 있다.

### 1. frame-wise causal calibration

먼저 각 branch의 출력을 비교 가능한 scale로 맞추기 위해 frame-wise causal calibration head를 둔다. 이 head는 각 time frame마다 두 가지 scalar를 출력한다고 볼 수 있다.

- $c_t$: 두 branch에 공통으로 작용하는 overall scale
- $\Delta_t$: 두 branch 사이의 상대적인 scale mismatch

즉

$$
c_t = q(z_{\le t}), \qquad
\Delta_t = h(z_{\le t})
$$

와 같이 공통 scale과 상대 scale을 causal 하게 예측하고,

$$
g^{bcs}_{t}
=
c_t\,\exp\left(-\frac{\Delta_t}{2}\right),
\qquad
g^{acs}_{t}
=
c_t\,\exp\left(\frac{\Delta_t}{2}\right)
$$

$$
\bar{S}^{bcs}_{f,t} = g^{bcs}_{t} S^{bcs}_{f,t}, \qquad
\bar{S}^{acs}_{f,t} = g^{acs}_{t} S^{acs}_{f,t}
$$

처럼 frame-wise gain을 주파수축 전체에 공유하여 적용한다. 이렇게 하면 calibration head는 특정 frequency bin을 선택적으로 조정하는 enhancement mask처럼 동작할 수 없고, frame-level loudness alignment만 담당하게 된다.

### 2. reliability-based fusion

그 다음 두 branch의 상대 reliability를 추정하고, 이를 정규화된 결합 weight로 바꾼다.

$$
r^{bcs}_{f,t} = \phi(u^{bcs}_{\le t}), \qquad
r^{acs}_{f,t} = \phi(u^{acs}_{\le t})
$$

$$
\alpha^{bcs}_{f,t}
=
\frac{r^{bcs}_{f,t}}{r^{bcs}_{f,t} + r^{acs}_{f,t} + \epsilon},
\qquad
\alpha^{acs}_{f,t}
=
\frac{r^{acs}_{f,t}}{r^{bcs}_{f,t} + r^{acs}_{f,t} + \epsilon}
$$

$$
\hat{S}_{f,t}
=
\alpha^{bcs}_{f,t}\,\bar{S}^{bcs}_{f,t}
+
\alpha^{acs}_{f,t}\,\bar{S}^{acs}_{f,t}
$$

즉 개선 모델의 핵심은 다음과 같다.

- calibration head는 frame-level 공통 scale과 상대 scale을 통해 두 branch를 비교 가능한 영역으로 옮긴다.
- fusion head는 그 위에서 두 branch의 상대 reliability를 추정한다.
- 최종 weight는 softmax 또는 normalized reliability처럼 해석된다.

이 구조는 기존의 "`acs`가 만든 gate 하나로 두 출력을 섞는 방식"보다 더 대칭적이며, minimum-variance 또는 posterior-weighted fusion에 더 가까운 해석을 제공한다.

## 각 요소의 의미

### 1. 왜 `bcs` 정보가 fusion에 들어가야 하는가

현재 구조에서는

$$
\alpha_{f,t} = g(m^{acs}_{f,t})
$$

이므로, 최종 결합은 두 branch의 결합인데 결합 비율을 정하는 정보는 사실상 한 branch에서만 온다.

반면 개선 모델에서는

$$
\alpha_{f,t} = g(m^{acs}_{f,t}, m^{bcs}_{f,t})
$$

또는 그보다 일반적으로

$$
r^{bcs}_{f,t} = \phi(z^{bcs}_{f,t}), \qquad
r^{acs}_{f,t} = \phi(z^{acs}_{f,t})
$$

처럼 두 branch 모두로부터 reliability를 만든다.

이 차이는 단순히 feature를 하나 더 넣는 문제가 아니다. 의미상으로는 `alpha`를 "`acs`에만 condition된 gate"에서 "`acs`와 `bcs`의 상대 reliability를 반영하는 weight"로 바꾸는 것이다.

### 2. minimum-variance 관점에서의 해석

만약 두 estimator의 오차가 unbiased이고 분산이 각각 $\sigma_b^2$, $\sigma_a^2$라면, 최소분산 선형 결합의 최적 weight는 inverse variance에 비례한다.

$$
r^{bcs}_{f,t} = \sigma_b^{-2}, \qquad
r^{acs}_{f,t} = \sigma_a^{-2}
$$

$$
\alpha^{bcs}_{f,t}
=
\frac{r^{bcs}_{f,t}}{r^{bcs}_{f,t} + r^{acs}_{f,t}},
\qquad
\alpha^{acs}_{f,t}
=
\frac{r^{acs}_{f,t}}{r^{bcs}_{f,t} + r^{acs}_{f,t}}
$$

따라서 개선 모델은 실제로 $\sigma^{-2}$를 직접 출력하지 않더라도, magnitude, mask, log-ratio, 또는 CNN이 만든 hidden feature를 reliability의 proxy로 사용하여 minimum-variance 계열의 fusion을 근사하는 것으로 해석할 수 있다. 여기서 calibration은 estimator scale을 맞추는 전처리 단계이고, reliability head는 그 위에서 inverse-variance-like weighting을 근사하는 단계라고 볼 수 있다.

### 3. 왜 raw magnitude를 그대로 confidence로 보면 조심해야 하는가

`bcs` magnitude가 크다고 해서 항상 speech quality가 좋다는 뜻은 아니다.

예를 들어 bone conduction은 다음과 같은 성질을 가질 수 있다.

- voiced 또는 저주파 대역은 강하게 보존한다.
- 고주파 성분이나 세밀한 spectral detail은 약하거나 왜곡될 수 있다.
- absolute magnitude가 크더라도 perceptual quality 또는 phonetic detail이 더 좋다고 단정할 수 없다.

따라서 raw magnitude 그 자체를 confidence로 해석하기보다는 다음과 같이 쓰는 편이 더 자연스럽다.

- 정규화된 magnitude
- ACS 대비 상대 비율
- log-domain difference
- CNN이 학습하는 reliability feature

### 4. 왜 calibration이 먼저 필요한가

두 branch를 섞을 때 weight의 합이 1이라고 해서 실제 기여도가 공정하게 분배되는 것은 아니다.

$$
\hat{S}_{f,t}
=
\alpha^{bcs}_{f,t} S^{bcs}_{f,t}
+
\alpha^{acs}_{f,t} S^{acs}_{f,t},
\qquad
\alpha^{bcs}_{f,t} + \alpha^{acs}_{f,t} = 1
$$

라고 하더라도, 만약

$$
|S^{bcs}_{f,t}| \gg |S^{acs}_{f,t}|
$$

이면, 동일한 weight를 주더라도 실제 출력은 magnitude가 큰 branch에 의해 지배될 수 있다.

즉 convex combination은 weight의 합을 1로 맞출 뿐이지, 두 estimator가 동일한 scale 위에서 비교되고 있다는 것을 보장하지는 않는다.

이 경우 `alpha`는 "누가 더 신뢰할 만한가"를 나타내기보다, branch 간 amplitude mismatch를 보정하는 역할까지 함께 떠맡게 된다. 따라서 `alpha`를 reliability weight로 해석하려면, 먼저 branch scale을 맞추는 calibration 과정이 필요하다.

### 5. 왜 calibration은 causal 하고 frame-wise 해야 하는가

실제 streaming 또는 low-latency 환경에서는 future frame을 사용할 수 없으므로, calibration head도 causal 해야 한다.

또한 여기서 calibration의 목적은 enhancement가 아니라 branch 간 loudness alignment이다. 이미 pretrained된 mapping, masking backbone이 enhancement를 대부분 수행하고 있다고 가정하면, calibration head는 fine-grained TF mask를 다시 학습하기보다 frame-level gain controller로 제한하는 편이 더 자연스럽다.

즉 공통 scale과 상대 scale은

$$
c_t = q(z_{\le t}), \qquad
\Delta_t = h(z_{\le t})
$$

처럼 현재 시점까지의 정보만으로 계산되어야 한다.

여기서 $z$는 frame-wise로 요약된 magnitude, mask, log-magnitude, log-ratio, short-term energy summary 등의 feature일 수 있고, $q$와 $h$는 작은 causal temporal layer로 구성되는 것이 자연스럽다. 그 결과로 얻어지는 gain

$$
g^{bcs}_{t}
=
c_t\,\exp\left(-\frac{\Delta_t}{2}\right),
\qquad
g^{acs}_{t}
=
c_t\,\exp\left(\frac{\Delta_t}{2}\right)
$$

은 양수여야 하므로 `sigmoid`, `softplus`, `exp` 등으로 parameterize하는 것이 자연스럽다. 중요한 점은 출력 차원이 시간축만 가져야 하며, 이를 주파수축 전체에 broadcast함으로써 calibration head가 enhancement mask처럼 작동하는 것을 구조적으로 막을 수 있다는 것이다.

### 6. 왜 공통 scale과 상대 scale로 분해하는가

$g^{bcs}$와 $g^{acs}$를 완전히 독립적으로 두면 자유도가 너무 커질 수 있다. 이 경우 calibration head가 사실상 또 하나의 enhancement 모듈처럼 동작하거나, gain과 alpha가 서로 역할을 나눠 가지며 식별성이 약해질 수 있다. 반대로 한 branch를 완전히 anchor로 고정하면, 그 branch 자체의 scale 변동을 충분히 다루기 어렵다.

그래서 더 해석이 깔끔한 방식은 calibration을 공통 scale과 상대 scale로 분해하는 것이다.

현재 frame의 전체적인 loudness 또는 global calibration을 담당하는 공통 scale을

$$
c_t = q(z_{\le t})
$$

두 branch 사이의 상대적 scale mismatch를 보정하는 변수를

$$
\Delta_t = h(z_{\le t})
$$

라고 두자. 그러면 gain을 다음과 같이 parameterize할 수 있다.

$$
g^{bcs}_{t}
=
c_t\,\exp\left(-\frac{\Delta_t}{2}\right),
\qquad
g^{acs}_{t}
=
c_t\,\exp\left(\frac{\Delta_t}{2}\right)
$$

이때 보정된 branch 출력은

$$
\bar{S}^{bcs}_{f,t}
=
c_t\,\exp\left(-\frac{\Delta_t}{2}\right) S^{bcs}_{f,t}
$$

$$
\bar{S}^{acs}_{f,t}
=
c_t\,\exp\left(\frac{\Delta_t}{2}\right) S^{acs}_{f,t}
$$

가 된다.

이 parameterization의 장점은 다음과 같다.

- $c_t$는 두 branch에 공통으로 작용하는 전체 scale 보정을 담당한다.
- $\Delta_t$는 `bcs`와 `acs` 사이의 상대적인 level mismatch만 조정한다.
- 두 branch 모두 scaling이 가능하므로, 어느 한쪽을 절대적인 anchor로 둘 필요가 없다.
- 완전 독립 gain 두 개를 두는 것보다 자유도가 작아서 해석이 더 안정적이다.
- 출력이 frame-wise scalar이므로 calibration head의 capacity가 낮고, pretrained backbone의 enhancement 기능을 침범하기 어렵다.

특히 상대 scale에 대한 해석은 여전히 유지된다. 공통 scale을 제외한 두 gain의 비는

$$
\frac{g^{acs}_{t}}{g^{bcs}_{t}} = \exp(\Delta_t)
$$

가 되므로, $\Delta_t$는 두 branch의 상대적 크기 차이를 직접 나타내는 변수로 볼 수 있다.

또한

$$
g^{bcs}_{t} \cdot g^{acs}_{t} = c_t^2
$$

이므로, 상대 scale 조정은 $\Delta_t$가 담당하고 전체적인 loudness 이동은 $c_t$가 담당한다고 해석할 수 있다.

이 parameterization의 해석은 다음처럼 분리된다.

- calibration head는 공통 scale과 상대 scale을 통해 두 branch를 비교 가능한 영역으로 옮긴다.
- fusion head는 그 위에서 어느 branch가 더 reliable한지를 결정한다.

또한 이 분해는 edge case 해석에도 유리하다. 예를 들어 `acs`가 매우 낮은 SNR로 인해 거의 collapse된 경우에는 calibration head가 이를 강하게 복구하려 하기보다, 제한된 범위 내에서 frame-level scale만 보정하고 실제 branch 선택은 reliability head가 담당하는 것이 자연스럽다. 즉 calibration은 scale alignment를 위한 약한 보정이어야 하고, branch failure의 처리는 fusion weight 쪽에서 이루어지는 것이 해석상 더 타당하다.

즉 calibration과 reliability fusion을 분리하면, 최종 `alpha`를 minimum-variance 계열의 posterior-like weight로 해석하기가 훨씬 쉬워진다.

### 7. calibration head의 추상적 구조

위 해석을 구현 관점으로 옮기면 calibration head는 고해상도 TF predictor가 아니라, 주파수축 정보를 먼저 요약한 뒤 시간축 sequence만 처리하는 작은 causal module로 설계하는 것이 적절하다.

가장 자연스러운 추상 구조는 다음과 같다.

1. 각 frame에서 `bcs`와 `acs`의 magnitude 또는 energy를 주파수축으로 요약한다.
2. `acs` mask의 frame 평균이나 frame variance 같은 confidence-like summary를 함께 만든다.
3. 이 frame-wise feature sequence를 작은 causal temporal encoder에 넣는다.
4. encoder 출력으로부터 $c_t$와 $\Delta_t$를 예측한다.

이때 temporal encoder는 복잡할 필요가 없다. calibration의 출력 차원이 시간축만 가지므로, 작은 causal 1D convolution stack 또는 매우 얕은 recurrent layer 정도면 충분하다. 중요한 것은 표현력을 키우는 것이 아니라, frame-level loudness trend와 branch 간 상대 scale mismatch를 안정적으로 추적하는 것이다.

## 정리

현재 `BAFNet`은 `acs`가 만든 mask에서 gate를 계산하는 비대칭 fusion 구조로 볼 수 있다. 반면 개선 모델은

1. causal calibration으로 branch scale을 먼저 정렬하고
2. reliability-based fusion으로 상대 신뢰도를 추정한 뒤
3. normalized weight로 두 branch를 결합하는

구조로 해석할 수 있다.

이때 calibration을 공통 scale $c_t$와 상대 scale $\Delta_t$로 분해하면,

- 두 branch 모두의 scale 변동을 허용할 수 있고
- 완전 독립 gain보다 자유도를 줄일 수 있으며
- frame-wise 출력만 갖도록 제한하여 calibration이 enhancement mask처럼 작동하는 것을 막을 수 있고
- 최종 fusion weight를 posterior-like reliability로 해석하기 쉬워진다.

따라서 이 문서의 개선 방향은 "`acs`-conditioned gate"를 "`causal calibration + reliability-based fusion`" 구조로 일반화하는 것으로 요약할 수 있다.

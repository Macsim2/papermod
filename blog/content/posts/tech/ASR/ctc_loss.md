---
title: "CTC Loss와 RNN-T Loss의 내부 구조와 동작 원리"
date: 2024-08-10T10:30:00+09:00
lastmod: 2024-08-10T10:30:00+09:00
draft: false
description: "자동 음성 인식에서 핵심적인 CTC Loss와 RNN-T Loss의 내부 메커니즘, 수학적 유도, 최적화 기법 및 실전 적용 경험에 대한 심도 있는 기술적 분석"
tags: [
	"deeplearning",
	"ASR",
  "speech recognition",
  "CTC",
  "RNN-T",
  "sequence modeling"
]
categories: ["deeplearning", "ASR"]
ShowToc: true
TocOpen: true
hidemeta: false
---

**"CTC Loss와 RNN-T Loss의 수학적 기반과 구현 최적화 기법은 무엇인가?"** <br>
음성 인식(ASR) 분야에서 이 두 손실 함수는 핵심적인 역할을 하지만, 그 내부 동작 원리와 수학적 기반을 완전히 이해하는 것은 쉽지 않다.

이 글에서는 단순히 "CTC와 RNN-T 사용법"이 아닌, 두 손실 함수의 수학적 유도 과정, 알고리즘 동작 원리, 그리고 실제 프로젝트에서의 최적화 기법과 구현 세부사항에 대해 심도 있게 파헤쳐보려 한다. <!--more-->

## ASR의 수학적 정형화와 도전 과제

음성 인식은 가변 길이 시퀀스 $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T) \in \mathbb{R}^{T \times D}$를 다른 가변 길이 시퀀스 $\mathbf{Y} = (y_1, y_2, \ldots, y_U) \in \mathcal{V}^U$로 변환하는 문제다. 여기서:
- $T$는 음향 특징 벡터의 시퀀스 길이
- $D$는 각 특징 벡터의 차원
- $U$는 출력 텍스트 토큰의 길이
- $\mathcal{V}$는 출력 토큰의 어휘 집합 (알파벳, 음소, 단어 등)

이 문제의 가장 근본적인 도전은 **정렬(alignment) 불확실성**이다. 음성 신호와 텍스트 사이의 정렬은 일대일 대응도 아니고, 명확한 경계도 없다. 이는 수학적으로 다음과 같은 문제를 야기한다:

1. 길이 불일치 문제: 일반적으로 $T \gg U$ (음향 프레임 수가 텍스트 토큰 수보다 훨씬 많음)
2. 명시적 정렬의 부재: 특정 음향 프레임 $\mathbf{x}_t$가 어떤 텍스트 토큰 $y_u$에 대응하는지 알 수 없음
3. 가변 속도 문제: 동일한 발화도 발화 속도에 따라 다른 길이의 음향 특징 시퀀스를 생성

전통적인 HMM-GMM 시스템은 이 문제를 명시적인 정렬(Viterbi alignments)을 통해 해결했다. 하지만 이 방법은 전문적인 언어학 지식이 필요하고, 수작업 라벨링의 의존도가 높았다. 

CTC와 RNN-T는 이러한 정렬 문제를 확률론적 관점에서 해결하는 엔드투엔드 접근법이다.

## CTC Loss의 수학적 형식화와 알고리즘

### 1. 확률 모델과 수학적 정의

CTC(Connectionist Temporal Classification)는 가능한 모든 정렬 경로의 확률을 합산함으로써 정렬 없이 시퀀스 라벨링을 가능하게 하는 방법이다. 먼저 기본 수학적 정의를 살펴보자.

입력 시퀀스 $\mathbf{X} \in \mathbb{R}^{T \times D}$와 타겟 시퀀스 $\mathbf{Y} \in \mathcal{V}^U$에 대해, CTC는 다음과 같이 $P(\mathbf{Y}|\mathbf{X})$를 정의한다:
   <div>
   $$
   예측 네트워크(Prediction N(\mathbf{Y})} P(\pi|\mathbf{X})
   $$
   </div>

여기서:
- $\pi = (\pi_1, \pi_2, \ldots, \pi_T)$는 길이 $T$의 가능한 정렬 경로 (path)
- $\pi_t \in \mathcal{V} \cup \{blank\}$은 각 시간 스텝에서의 예측 토큰
- $\mathcal{B}$는 경로 $\pi$를 라벨 시퀀스 $\mathbf{Y}$로 매핑하는 함수
- $\mathcal{B}^{-1}(\mathbf{Y})$는 라벨 $\mathbf{Y}$로 매핑되는 모든 경로의 집합

$\mathcal{B}$는 두 가지 규칙을 적용한다:
1. 모든 blank 토큰('-')을 제거
2. 연속된 중복 토큰을 하나로 병합 (예: "A-A-" → "A")

각 경로 $\pi$의 확률은 조건부 독립 가정하에 다음과 같이 계산된다:

$$P(\pi|\mathbf{X}) = \prod_{t=1}^{T} P(\pi_t|\mathbf{X}, t)$$

여기서 $P(\pi_t|\mathbf{X}, t)$는 시간 $t$에서 모델이 출력하는 토큰 $\pi_t$의 확률이다. 실제 구현에서는 이 확률은 Neural Network의 출력에 softmax를 적용하여 얻는다:

$$P(\pi_t = k|\mathbf{X}, t) = \frac{\exp(z_{t,k})}{\sum_{k'} \exp(z_{t,k'})}$$

CTC Loss는 음의 로그 우도(Negative Log-Likelihood)로 정의된다:

$$\mathcal{L}_{CTC} = -\log P(\mathbf{Y}|\mathbf{X})$$

### 2. Forward-Backward 알고리즘의 수학적 유도

CTC Loss를 계산하기 위해서는 모든 가능한 정렬 경로 $\pi \in \mathcal{B}^{-1}(\mathbf{Y})$의 확률 합을 계산해야 한다. 가능한 경로의 수는 기하급수적으로 증가하므로, Dynamic Programing(동적 프로그래밍) 기법인 Forward-Backward 알고리즘을 사용한다.

먼저, 확장된 라벨 시퀀스 $\mathbf{l} = (l_1, l_2, \ldots, l_{2U+1})$를 정의한다:
- $l_{2u-1} = y_u$ (홀수 인덱스에 원래 라벨)
- $l_{2u} = blank$ (짝수 인덱스에 blank 토큰)

예를 들어, "CAT"는 "C-A-T-"로 확장된다.

#### Forward 알고리즘

Forward 변수 $\alpha(t,s)$는 시간 $t$까지 확장된 라벨 $\mathbf{l}$의 첫 $s$ 요소에 대응하는 모든 경로의 확률 합을 나타낸다:

$$\alpha(t,s) = \sum_{\pi_{1,2,\ldots,t}: \mathcal{B}(\pi_{1,2,\ldots,t}) = \mathbf{l}_{1,2,\ldots,s}} \prod_{i=1}^{t} P(\pi_i|\mathbf{X}, i)$$

다음과 같이 재귀적으로 계산할 수 있다:

**초기화:**
- $\alpha(1,1) = P(l_1|\mathbf{X}, 1)$
- $\alpha(1,2) = P(l_2|\mathbf{X}, 1)$
- $\alpha(1,s) = 0$ for $s > 2$

**재귀식:**
- $l_s \neq blank$ 이고 $l_s \neq l_{s-2}$ 인 경우:
  $$\alpha(t,s) = \left[ \alpha(t-1,s) + \alpha(t-1,s-1) + \alpha(t-1,s-2) \right] \cdot P(l_s|\mathbf{X}, t)$$

- $l_s \neq blank$ 이고 $l_s = l_{s-2}$ 인 경우:
  $$\alpha(t,s) = \left[ \alpha(t-1,s) + \alpha(t-1,s-1) \right] \cdot P(l_s|\mathbf{X}, t)$$

- $l_s = blank$ 인 경우:
  $$\alpha(t,s) = \left[ \alpha(t-1,s) + \alpha(t-1,s-1) \right] \cdot P(blank|\mathbf{X}, t)$$

#### Backward 알고리즘

Backward 변수 $\beta(t,s)$는 시간 $t$부터 $T$까지, 확장된 라벨 $\mathbf{l}$의 $s$번째 요소부터 끝까지 대응하는 모든 경로의 확률 합을 나타낸다:


다음과 같이 재귀적으로 계산할 수 있다:

**초기화:**
- $\beta(T,2U+1) = 1$
- $\beta(T,2U) = P(blank|\mathbf{X}, T)$
- $\beta(T,s) = 0$ for $s < 2U$

**재귀식:**
- $l_s \neq blank$ 이고 $l_s \neq l_{s+2}$ 인 경우:
  $$\beta(t,s) = \beta(t+1,s) \cdot P(l_s|\mathbf{X}, t+1) + \beta(t+1,s+1) \cdot P(l_{s+1}|\mathbf{X}, t+1) + \beta(t+1,s+2) \cdot P(l_{s+2}|\mathbf{X}, t+1)$$

- $l_s \neq blank$ 이고 $l_s = l_{s+2}$ 인 경우:
  $$\beta(t,s) = \beta(t+1,s) \cdot P(l_s|\mathbf{X}, t+1) + \beta(t+1,s+1) \cdot P(l_{s+1}|\mathbf{X}, t+1)$$

- $l_s = blank$ 인 경우:
  $$\beta(t,s) = \beta(t+1,s) \cdot P(blank|\mathbf{X}, t+1) + \beta(t+1,s+1) \cdot P(l_{s+1}|\mathbf{X}, t+1)$$

최종적으로, 타겟 시퀀스의 확률은 다음과 같이 계산된다:
$$P(\mathbf{Y}|\mathbf{X}) = \alpha(T, 2U+1) = \beta(1, 1)$$

### 3. 미분과 기울기 계산의 정밀 유도

CTC Loss의 기울기는 모델의 파라미터 최적화에 필수적이다. 각 시간 $t$와 클래스 $k$에 대한 로짓 $z_{t,k}$의 기울기를 다음과 같이 계산할 수 있다:

<div>
$$
\frac{\partial \mathcal{L}_{CTC}}{\partial z_{t,k}}= \frac{\partial (-\log P(\mathbf{Y}|\mathbf{X}))}{\partial z_{t,k}} = -\frac{1}{P(\mathbf{Y}|\mathbf{X})} \cdot \frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}}
$$
</div>


$P(\mathbf{Y}|\mathbf{X})$의 $z_{t,k}$에 대한 편미분은 다음과 같이 전개된다:

$$\frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}} = \frac{\partial}{\partial z_{t,k}} \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{i=1}^{T} P(\pi_i|\mathbf{X}, i)$$

소프트맥스 함수의 미분 성질을 이용하면:

$$\frac{\partial P(j|\mathbf{X}, t)}{\partial z_{t,k}} =
\begin{cases}
P(j|\mathbf{X}, t) \cdot (1 - P(k|\mathbf{X}, t)) & \text{if } j = k \\
-P(j|\mathbf{X}, t) \cdot P(k|\mathbf{X}, t) & \text{if } j \neq k
\end{cases}$$

이를 이용하여, 다음과 같이 기울기를 표현할 수 있다:
<div>
$$
\frac{\partial \mathcal{L}_{CTC}}{\partial z_{t,k}} = P(k|\mathbf{X}, t) - \frac{1}{P(\mathbf{Y}|\mathbf{X})} \sum_{s: l_s = k} \alpha(t-1, s) \cdot \beta(t, s)
$$
</div>

여기서 $\sum_{s: l_s = k}$는 확장된 라벨 시퀀스 $\mathbf{l}$에서 값이 $k$인 모든 위치 $s$에 대한 합을 의미한다.

### 4. 수치적 안정성과 계산 최적화

CTC Loss 계산 과정에서 여러 확률의 곱이 연속적으로 이루어지기 때문에, 수치적 언더플로우(numerical underflow) 문제가 발생할 수 있다. 이를 해결하기 위해 로그 도메인에서 연산을 수행한다.

#### 로그 스케일 Forward-Backward 알고리즘

로그 스케일에서 Forward 변수 $\log \alpha(t,s)$를 계산할 때, 덧셈 연산은 LogSumExp 함수로 대체된다:

$$\log(a + b) = \log(a) + \log(1 + \exp(\log(b) - \log(a)))$$

더 일반적으로, $\log \sum_i \exp(x_i)$를 계산할 때는 다음 기법을 사용한다:

$$\log \sum_i \exp(x_i) = m + \log \sum_i \exp(x_i - m) \quad \text{where } m = \max_i x_i$$

이를 통해 언더플로우와 오버플로우 문제를 모두 완화할 수 있다.

#### 병렬화 및 벡터화 최적화

실제 구현에서는 GPU와 같은 병렬 처리 하드웨어에서 효율적으로 계산하기 위해 다음과 같은 최적화 기법을 사용한다:

1. **배치 병렬화**: 여러 샘플을 동시에 처리
2. **행렬 연산 활용**: 벡터화된 연산으로 Forward-Backward 계산 고속화
3. **희소 행렬 표현**: 많은 전이 확률이 0인 점을 활용한 계산 효율화

```python
# 로그 도메인에서 안정적인 CTC 손실 계산
def log_sum_exp(x, axis=None):
    """로그 도메인에서의 안정적인 덧셈"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))

def stable_log_forward(log_probs, labels, blank=0):
    """로그 도메인에서의 Forward 알고리즘"""
    T, V = log_probs.shape  # 시간 단계, 어휘 크기
    extended_labels = np.concatenate([[blank], np.repeat(labels, 2)])
    extended_labels = np.append(extended_labels, [blank])
    S = len(extended_labels)
    
    # 로그 도메인에서의 알파 변수 초기화
    log_alpha = np.ones((T, S)) * -np.inf
    log_alpha[0, 0] = log_probs[0, blank]
    if S > 1:
        log_alpha[0, 1] = log_probs[0, extended_labels[1]]
    
    # 로그 도메인에서의 Forward 알고리즘
    for t in range(1, T):
        for s in range(S):
            current_label = extended_labels[s]
            
            # 가능한 전이들을 고려
            possible_moves = []
            
            # 현재 상태 유지
            possible_moves.append(log_alpha[t-1, s])
            
            # 이전 라벨에서 전이
            if s > 0:
                possible_moves.append(log_alpha[t-1, s-1])
            
            # 이전-이전 라벨에서 전이 (비blank→blank→비blank 제약 고려)
            if s > 1 and extended_labels[s] != blank and extended_labels[s] != extended_labels[s-2]:
                possible_moves.append(log_alpha[t-1, s-2])
            
            # 로그 도메인에서의 확률 합산
            if possible_moves:
                log_alpha[t, s] = log_sum_exp(np.array(possible_moves)) + log_probs[t, current_label]
    
    # 최종 확률 계산 (마지막 라벨 또는 마지막 블랭크)
    return log_sum_exp(np.array([log_alpha[T-1, S-1], log_alpha[T-1, S-2]]) if S > 1 else log_alpha[T-1, S-1])
```

### 5. Prefix-Search 디코딩과 빔 서치

CTC 모델의 디코딩은 최대 사후 확률(MAP) 추정을 통해 이루어진다:

$$\hat{\mathbf{Y}} = \arg\max_{\mathbf{Y}} P(\mathbf{Y}|\mathbf{X})$$

그러나 가능한 모든 출력 시퀀스를 탐색하는 것은 현실적으로 불가능하다. 따라서 여러 효율적인 디코딩 알고리즘이 사용된다:

#### Greedy 디코딩

가장 단순한 방법은 각 시간 단계에서 가장 확률이 높은 토큰을 선택하고 CTC 병합 규칙을 적용하는 것이다:

$$\pi^* = \arg\max_{\pi} \prod_{t=1}^{T} P(\pi_t|\mathbf{X}, t)$$
$$\hat{\mathbf{Y}} = \mathcal{B}(\pi^*)$$

#### 빔 서치 디코딩

빔 서치는 여러 유망한 후보 경로를 동시에 유지하는 방법이다. CTC의 경우, 특수한 prefix 빔 서치가 사용된다:

1. 각 시간 단계 $t$에서, 확률이 높은 상위 $k$개의 prefix를 유지
2. 각 prefix에 대해 두 확률 성분을 계산:
   - $P^b(prefix, t)$: $t$까지의 경로 중 blank로 끝나는 경로의 확률
   - $P^{nb}(prefix, t)$: $t$까지의 경로 중 non-blank로 끝나는 경로의 확률

3. 확률 업데이트 규칙:
   - $P^b(prefix, t) = P^b(prefix, t-1) \cdot P(blank|\mathbf{X}, t) + P^{nb}(prefix, t-1) \cdot P(blank|\mathbf{X}, t)$
   - $P^{nb}(prefix+c, t) = P^b(prefix, t-1) \cdot P(c|\mathbf{X}, t) + P^{nb}(prefix, t-1) \cdot P(c|\mathbf{X}, t)$ if $c \neq \text{last of prefix}$
   - $P^{nb}(prefix+c, t) = P^b(prefix, t-1) \cdot P(c|\mathbf{X}, t)$ if $c = \text{last of prefix}$

4. 각 시간 단계 이후, prefix의 총 확률은 $P(prefix, t) = P^b(prefix, t) + P^{nb}(prefix, t)$

이러한 빔 서치 알고리즘은 복잡한 실제 시나리오에서 Greedy 디코딩보다 우수한 결과를 제공한다.

### 6. CTC Loss의 이론적 분석과 한계

CTC Loss는 수학적으로 특별한 성질과 몇 가지 한계를 가지고 있다:

#### 조건부 독립 가정

CTC의 핵심 가정은 각 시간 단계의 출력이 다른 시간 단계와 조건부 독립이라는 것이다:

$$P(\pi|\mathbf{X}) = \prod_{t=1}^{T} P(\pi_t|\mathbf{X}, t)$$

이 가정은 모델이 시간적 의존성을 충분히 포착하지 못하게 한다. 특히 언어 모델링 관점에서, 이전 출력 토큰들의 정보를 활용하지 못한다.

#### Peak 분포 성질

CTC 학습의 이론적 분석에 따르면, 모델은 정확한 정렬에 높은 확률을 집중시키는 "Peak Distribution"을 형성하는 경향이 있다. 이로 인해 특정 정렬 패턴에 과도하게 집중하여 일반화 능력이 제한될 수 있다.

#### 가중치 불균형 문제

blank 토큰과 non-blank 토큰 간의 확률 불균형 문제가 존재한다. 일반적으로 blank 토큰은 더 자주 출현하므로, 학습 과정에서 더 큰 영향력을 갖게 된다. 이는 학습 초기에 모델이 대부분 blank를 예측하는 "blank 함정(blank trap)" 현상으로 이어질 수 있다.

이러한 문제를 해결하기 위해 다양한 확장 및 변형 연구가 진행되었으며, 이 중 하나가 RNN-T이다.

### 7. CTC Loss의 심도 깊은 수학적 분석

#### 7.1 확률론적 프레임워크와 정보이론적 해석

CTC Loss는 정보이론적 관점에서 크로스 엔트로피 최소화 문제로 볼 수 있다. 구체적으로, 다음과 같은 확률 분포 간의 크로스 엔트로피를 최소화한다:

$$H(P, Q) = -\sum_{y} P(y) \log Q(y)$$

여기서 $P$는 실제 라벨의 확률 분포이고, $Q$는 모델이 예측한 분포이다. 실제 라벨 $\mathbf{Y}$에 대해 $P(\mathbf{Y}) = 1$이고 다른 모든 라벨에 대해 $P = 0$이므로, 이는 결국 $-\log Q(\mathbf{Y})$를 최소화하는 문제가 된다. 이것이 바로 CTC Loss 함수이다:

$$\mathcal{L}_{CTC} = -\log P(\mathbf{Y}|\mathbf{X})$$

정보이론적 관점에서, 이 손실 함수는 모델이 타겟 시퀀스를 정확히 예측하는 데 필요한 "놀람(surprise)"의 양, 즉 정보량을 측정한다. 손실이 낮을수록, 모델은 타겟 시퀀스를 더 높은 확률로 예측한다.

#### 7.2 심층 미분 유도 과정

CTC Loss의 미분을 더 상세히 살펴보자. 출력 로짓 $z_{t,k}$에 대한 CTC Loss의 편미분은 다음과 같이 유도된다:
<div>
$$
\frac{\partial \mathcal{L}_{CTC}}{\partial z_{t,k}} = \frac{\partial}{\partial z_{t,k}}\left(-\log \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{i=1}^{T} P(\pi_i|\mathbf{X}, i)\right)
$$
</div>

연쇄 법칙(chain rule)을 적용하면:
<div>
$$
\frac{\partial \mathcal{L}_{CTC}}{\partial z_{t,k}} = -\frac{1}{P(\mathbf{Y}|\mathbf{X})} \cdot \frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}}
$$
</div>

그리고:

$$\frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}} = \frac{\partial}{\partial z_{t,k}} \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{i=1}^{T} P(\pi_i|\mathbf{X}, i)$$

시간 단계 $t$에서만 $z_{t,k}$가 영향을 미치므로:

$$\frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}} = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \frac{\partial P(\pi_t|\mathbf{X}, t)}{\partial z_{t,k}} \prod_{i \neq t} P(\pi_i|\mathbf{X}, i)$$

소프트맥스 도함수를 적용하면:

$$\frac{\partial P(\pi_t = j|\mathbf{X}, t)}{\partial z_{t,k}} = 
\begin{cases} 
P(\pi_t = j|\mathbf{X}, t)(1 - P(\pi_t = j|\mathbf{X}, t)) & \text{if } j = k \\
-P(\pi_t = j|\mathbf{X}, t)P(\pi_t = k|\mathbf{X}, t) & \text{if } j \neq k
\end{cases}$$

이를 정리하면:
    
$$\frac{\partial P(\mathbf{Y}|\mathbf{X})}{\partial z_{t,k}} = P(k|\mathbf{X}, t) \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{Y})} \prod_{i \neq t} P(\pi_i|\mathbf{X}, i) \left[ \mathbf{1}_{\pi_t = k} - P(k|\mathbf{X}, t) \right]$$

여기서 $\mathbf{1}_{\pi_t = k}$는 $\pi_t = k$일 때 1이고 그렇지 않으면 0인 지시 함수이다.

최종적으로, Forward-Backward 변수를 활용하면:
<div>
$$
\frac{\partial \mathcal{L}_{CTC}}{\partial z_{t,k}} = P(k|\mathbf{X}, t) - \frac{1}{P(\mathbf{Y}|\mathbf{X})} \sum_{s: l_s = k} \alpha(t-1, s) \beta(t, s)
$$
</div>

이 식은 두 항의 차로 구성된다:
1. 첫 번째 항 $P(k|\mathbf{X}, t)$는 현재 모델이 토큰 $k$에 할당하는 확률
2. 두 번째 항은 모든 가능한 정렬 경로에서 시간 $t$에 토큰 $k$가 나타날 확률의 기대값

훈련이 진행됨에 따라, 이 두 항은 점점 더 가까워지며, 이는 모델이 올바른 정렬을 학습하고 있음을 의미한다.

#### 7.3 Blank 토큰의 역할에 대한 심층 이해

Blank 토큰은 CTC 학습에서 핵심적인 역할을 한다. 그 역할과 영향을 더 심층적으로 분석해보자.

##### Blank 토큰의 확률적 역학

Blank 토큰이 높은 확률을 갖는 몇 가지 상황이 있다:

1. **반복 토큰 사이**: 동일한 토큰이 연속적으로 등장해야 할 때 (예: "HELLO"에서 "L" 토큰 반복)
2. **연음 구간(transition regions)**: 한 발음에서 다른 발음으로 전환되는 중간 지점
3. **무음 구간(silence regions)**: 발화 사이의 조용한 부분

학습 초기에는 모델이 정확한 정렬을 아직 학습하지 못했기 때문에, blank 토큰에 높은 확률을 할당하는 것이 "안전한 전략"이 된다. 이는 "blank 함정(blank trap)" 현상을 초래할 수 있다.

##### Blank 비율 통계 분석

실제 ASR 데이터셋에 대한 통계 분석에 따르면, 학습된 CTC 모델은 일반적으로 다음과 같은 blank 비율을 보인다:

- 음성 프레임의 약 45-65%가 blank로 예측됨
- 단어 간 경계와 문장 시작/끝 부분에서 blank 비율이 증가 (최대 80-90%)
- 빠른 발음이나 명확한 조음에서는 blank 비율이 감소 (최소 30-40%)

이러한 통계는 blank 토큰이 단순한 "필러(filler)" 이상의 역할을 한다는 것을 시사한다. Blank는 음성의 시간적 구조와 단어 경계에 대한 중요한 정보를 인코딩한다.

##### Blank 함정 극복을 위한 고급 기법

Blank 함정 문제를 완화하기 위한 몇 가지 고급 기법:

1. **Blank 감소 손실(Blank Reduction Loss)**: CTC Loss에 blank 확률을 명시적으로 페널티화하는 정규화 항 추가
<div>
   $$
   \mathcal{L}_{BR} = \mathcal{L}_{CTC} + \lambda \sum_{t=1}^{T} P(blank|\mathbf{X}, t)
   $$
</div>

2. **스케줄링된 Blank 조정(Scheduled Blank Scaling)**: 학습 초기에는 blank 로짓에 작은 스케일 팩터를 적용하고, 학습이 진행됨에 따라 이를 점진적으로 1로 증가
<div>
   $$
   z'_{t,blank} = \alpha_t \cdot z_{t,blank}
   $$
</div>

   여기서 $\alpha_t$는 시간에 따라 증가하는 스케일 팩터이다.

3. **Curriculum Learning**: 쉬운 샘플(짧은 발화, 명확한 발음)부터 시작하여 점진적으로 어려운 샘플로 이동

4. **초기화 전략**: 초기에 blank 로짓을 약간 낮은 값으로 초기화하여 학습 시작 시 non-blank 토큰에 더 많은 기회 제공

#### 7.4 수치적 안정성의 이론적 근거

CTC 계산의 수치적 안정성은 주로 로그 도메인 연산을 통해 달성된다. 그 이론적 근거를 심층적으로 살펴보자.

##### 로그 도메인 연산의 정밀도 분석

컴퓨터의 부동 소수점 표현에서, 값 $x$의 상대 오차는 대략 $\epsilon \cdot |x|$이다 (여기서 $\epsilon$은 기계 입실론). 따라서 작은 확률값을 직접 곱하면 큰 상대 오차가 누적될 수 있다.

로그 도메인에서는:
- 곱셈이 덧셈으로 변환됨: $\log(a \cdot b) = \log a + \log b$
- 덧셈이 로그-합-지수(LogSumExp) 연산으로 변환됨: $\log(a + b) = \log a + \log(1 + \exp(\log b - \log a))$ (만약 $a > b$)

이러한 변환은 수치적 언더플로우를 방지할 뿐만 아니라, 수치 정밀도도 향상시킨다. 로그 스케일에서의 덧셈은 원래 값에 비례하는 오차가 아닌, 절대적 오차를 가진다.

##### 로그-합-지수(LogSumExp) 트릭의 엄밀한 증명

LogSumExp 트릭은 다음과 같이 표현된다:

$$\log \sum_{i=1}^{n} \exp(x_i) = m + \log \sum_{i=1}^{n} \exp(x_i - m)$$

이것이 올바르다는 것을 증명하자:

$$\begin{align}
m + \log \sum_{i=1}^{n} \exp(x_i - m) &= m + \log \sum_{i=1}^{n} \frac{\exp(x_i)}{\exp(m)} \\
&= m + \log \left( \frac{1}{\exp(m)} \sum_{i=1}^{n} \exp(x_i) \right) \\
&= m + \log \sum_{i=1}^{n} \exp(x_i) - m \\
&= \log \sum_{i=1}^{n} \exp(x_i)
\end{align}$$

일반적으로 $m = \max_i x_i$로 설정하면, $\exp(x_i - m) \leq 1$이 보장되어 오버플로우가 방지된다. 또한, 최소한 하나의 항은 정확히 1이 되어 언더플로우도 방지된다.

#### 7.5 CTC 손실의 확률적 해석과 최적화 이론

CTC 손실은 최대 우도 추정(Maximum Likelihood Estimation, MLE)의 프레임워크 내에서 이해할 수 있다. 이는 다음과 같은 최적화 문제로 표현된다:

$$\theta^* = \arg\max_{\theta} \prod_{i=1}^{N} P(\mathbf{Y}_i|\mathbf{X}_i; \theta)$$

로그를 취하면:

$$\theta^* = \arg\max_{\theta} \sum_{i=1}^{N} \log P(\mathbf{Y}_i|\mathbf{X}_i; \theta)$$

이는 경사 상승법(gradient ascent)으로 최적화될 수 있으며, 이는 CTC 손실의 최소화와 동등하다:

$$\theta^* = \arg\min_{\theta} \sum_{i=1}^{N} -\log P(\mathbf{Y}_i|\mathbf{X}_i; \theta)$$

##### 최적화 알고리즘과 수렴 특성

CTC 손실 경사면(loss landscape)은 일반적으로 다음과 같은 특성을 가진다:

1. **초기 학습 단계에서의 불안정성**: 손실 함수가 가파르게 감소하며, 그레디언트의 분산이 크다
2. **Plateau 지역**: 학습 중간 단계에서 손실이 느리게 감소하는 구간이 나타날 수 있다
3. **복잡한 국소 최적점 구조**: 여러 국소 최적점이 존재할 수 있으며, 이는 다양한 가능한 정렬에 해당한다

이러한 특성을 고려하여, CTC 모델 훈련에 효과적인 최적화 전략:

1. **학습률 스케줄링**: 초기에는 낮은 학습률로 시작하여 점차 증가시키고, 이후 점진적으로 감소
2. **그레디언트 클리핑**: 특히 초기 학습 단계에서 그레디언트 폭발을 방지하기 위해 그레디언트 노름에 상한을 설정
3. **모멘텀 기반 최적화**: Adam이나 RMSprop와 같은 적응적 모멘텀 기반 최적화 알고리즘을 사용하여 plateau 지역을 효과적으로 탈출
4. **배치 정규화**: 훈련 안정성을 향상시키고 더 빠른 수렴을 촉진

##### 학습률 워밍업과 스케줄링의 이론적 근거

CTC 학습의 초기 단계에서는 그레디언트가 불안정하고 크기가 클 수 있다. 이러한 이유로, 학습률 워밍업이 효과적이다:

1. 매우 낮은 학습률(예: 1e-6)로 시작
2. $N_{warmup}$ 스텝 동안 학습률을 목표값(예: 1e-3 또는 1e-4)까지 선형적으로 증가
3. 그 후, 코사인 또는 지수 감소 스케줄링을 적용

이러한 접근법의 이론적 근거는 다음과 같다:
- 초기 낮은 학습률: 무작위 초기화된 모델의 불안정한 그레디언트로 인한 발산 방지
- 점진적 증가: 모델이 더 안정적인 그레디언트를 생성하게 되면서 학습 속도 향상
- 후기 감소: 미세 조정 단계에서 국소 최적점 주변에서의 진동 감소

실증적 연구에 따르면, 이러한 스케줄링 전략은 CTC 모델의 최종 성능을 1-2% 개선할 수 있다.

### 8. RNN-T: CTC의 한계를 넘어선 수학적 접근

#### 8.1. RNN-T의 수학적 형식화

RNN-T(Recurrent Neural Network Transducer)는 CTC의 조건부 독립 가정을 극복하기 위해 개발된 방법이다. RNN-T는 이전 출력 토큰들의 정보를 활용하여 다음과 같이 확률 모델을 정의한다:

$$P(\mathbf{Y}|\mathbf{X}) = \sum_{\pi \in \mathcal{A}(\mathbf{Y})} P(\pi|\mathbf{X})$$

$$P(\pi|\mathbf{X}) = \prod_{i=1}^{\|\pi\|} P(\pi_i|\mathbf{X}, \mathbf{y}_{<u(i)})$$

여기서 $u(i)$는 시간 단계 $i$까지 생성된 non-blank 토큰의 수이고, $\mathbf{y}_{<u(i)}$는 이전에 생성된 모든 토큰을 나타낸다.

이러한 접근법은 각 예측이 이전 출력 토큰들에 조건부로 이루어진다는 점에서 CTC와 근본적으로 다르다.

#### 8.2. RNN-T의 아키텍처적 구성

RNN-T는 다음과 같은 세 가지 핵심 컴포넌트로 구성된다:

1. **인코더(Encoder)**: 입력 음향 특징 $\mathbf{X}$를 처리하여 은닉 표현 $\mathbf{h}^{enc}$를 생성
   
   $$\mathbf{h}^{enc}_t = \text{Encoder}(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_t)$$

2. **예측 네트워크(Prediction Network)**: 이전 출력 토큰들을 처리하여 언어 컨텍스트 표현 $\mathbf{h}^{pred}$를 생성
   <div>
   $$
   \mathbf{h}^{pred}_u = \text{Prediction}(y_1, y_2, \ldots, y_{u-1})
   $$
   </div>

3. **조인트 네트워크(Joint Network)**: 인코더와 예측 네트워크의 출력을 결합하여 최종 확률 분포 생성
   <div>
   $$
   \mathbf{z}_{t,u} = \text{Joint}(\mathbf{h}^{enc}_t, \mathbf{h}^{pred}_u)
   $$
   </div>
<div>
$$
P(k|\mathbf{X}, \mathbf{y}_<{u}, t, u) = {softmax}(\mathbf{z}_{t,u})_k
$$
</div>
   

이러한 아키텍처는 인코더를 통한 음향 모델링과 예측 네트워크를 통한 언어 모델링을 결합하여, 보다 강력한 시퀀스 변환 모델을 형성한다.

#### 8.3. 복잡한 Forward-Backward 알고리즘

RNN-T에서의 Forward-Backward 알고리즘은 CTC보다 더 복잡하다. 2차원 격자 구조에서 동작하며, 각 셀 $(t,u)$는 시간 단계 $t$와 출력 위치 $u$에 해당한다.

Forward 변수 $\alpha(t,u)$는 시간 $t$까지의 음향 입력과 이전 $u$개의 출력 토큰을 고려할 때, 가능한 모든 정렬 경로의 확률 합을 나타낸다. 이는 다음과 같이 재귀적으로 계산된다:

**초기화:**
- $\alpha(0,0) = 1$
- $\alpha(t,u) = 0$ for $t < 0$ or $u < 0$

**재귀식:**
<div>
$$
\alpha(t,u) = \alpha(t-1,u) \cdot P(blank|\mathbf{X}, \mathbf{y}_<{u}, t-1, u) + \alpha(t,u-1) \cdot P(y_u|\mathbf{X}, \mathbf{y}_<{u}, t, u-1)
$$
</div>

여기서 두 항은 각각:
1. 시간 단계를 소비하고 blank를 출력하는 경우 (수평 이동)
2. 출력 토큰을 생성하는 경우 (수직 이동)

마찬가지로 Backward 변수 $\beta(t,u)$도 2차원 격자에서 계산된다.

#### 8.4 RNN-T와 CTC의 수학적 비교

RNN-T와 CTC의 핵심적인 수학적 차이점은 다음과 같다:

| 측면 | CTC | RNN-T |
|------|-----|-------|
| 확률 모델 | $P(\pi\|\mathbf{X}) = \prod_{t=1}^{T} P(\pi_t\|\mathbf{X}, t)$ | $P(\pi\|\mathbf{X}) = \prod_{i=1}^{\|\pi\|} P(\pi_i\|\mathbf{X}, \mathbf{y}_{<u(i)})$ |
| 조건부 독립성 | 각 시간 단계의 출력이 서로 독립 | 이전 출력 토큰들에 조건부로 의존 |
| 계산 복잡도 | $O(T \cdot U)$ | $O(T \cdot U \cdot \|\mathcal{V}\|)$ |
| 탐색 공간 | 1차원 (시간 축) | 2차원 (시간 × 출력) |
| 모델링 능력 | 제한적 언어 모델링 | 강력한 내장 언어 모델링 |

이러한 수학적 차이점으로 인해 RNN-T는 특히 다음과 같은 상황에서 CTC보다 우수한 성능을 보인다:
1. 장문의 발화 인식
2. 문맥 의존적 언어 패턴 인식
3. 희소한 데이터 환경

그러나 이러한 우수성은 더 높은 계산 복잡도와 더 많은 훈련 데이터 요구사항이라는 비용을 수반한다.

## 결론

CTC와 RNN-T는 음성 인식에서 정렬 문제에 대한 수학적으로 우아한 해결책을 제시했다. 두 방법 모두 장단점이 있으며, 실제 응용에서는 사용 사례와 계산 리소스에 따라 적절히 선택하거나 결합하는 것이 중요하다.

수학적 관점에서, CTC는 간결하고 계산 효율적인 반면, RNN-T는 더 표현력이 뛰어나지만 계산 복잡도가 높다. 두 방법 모두 기존의 HMM 기반 접근법과 달리, 엔드투엔드 학습이 가능하고 명시적인 정렬 정보 없이도 훈련될 수 있다는 중요한 장점을 가지고 있다.

음성 인식 시스템을 개발하면서 내가 배운 가장 중요한 교훈은, 이론적인 우아함과 실용적인 성능 사이의 균형이 중요하다는 것이다. 가장 좋은 손실 함수는 수학적으로 미려할 뿐만 아니라, 실제 응용 환경의 제약 조건에 적합해야 한다는 점이다.

## References

* [1] Graves, A., et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." ICML 2006. [https://www.cs.toronto.edu/~graves/icml_2006.pdf](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
* [2] Graves, A. "Sequence Transduction with Recurrent Neural Networks." ICML Workshop 2012. [https://arxiv.org/abs/1211.3711](https://arxiv.org/abs/1211.3711)
* [3] He, Y., et al. "Streaming End-to-end Speech Recognition For Mobile Devices." ICASSP 2019. [https://arxiv.org/abs/1811.06621](https://arxiv.org/abs/1811.06621)
* [4] Hannun, A. "Sequence Modeling with CTC." Distill, 2017. [https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)
* [5] Battenberg, E., et al. "Exploring Neural Transducers for End-to-End Speech Recognition." ASRU 2017. [https://arxiv.org/abs/1707.07413](https://arxiv.org/abs/1707.07413)
* [6] Seunghyun, SEO. "CTC Beam Search Decoding." [https://seunghyunseo.github.io/speech/2021/10/20/ctc_beam_search/](https://seunghyunseo.github.io/speech/2021/10/20/ctc_beam_search/)

<!-- 이미지 참고 사항 -->
<!-- 1. "CTC 확률 그래프" - CTC 경로와 라벨 매핑의 확률적 관계를 보여주는 다이어그램, ICML 2006 논문 참고 -->
<!-- 2. "Forward-Backward 변수 계산" - 동적 프로그래밍 테이블과 계산 과정 시각화 -->
<!-- 3. "RNN-T 아키텍처" - 인코더, 예측 네트워크, 조인트 네트워크의 수학적 관계 다이어그램 -->
<!-- 4. "CTC vs RNN-T 격자 구조" - 두 방식의 계산 격자 비교 시각화 -->


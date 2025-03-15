---
title: "Viterbi in HMM-GMM"
date: 2025-03-15T01:17:26+09:00
lastmod: 2025-03-15T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- tech
tags: 
- ASR
- viterbi
description: "about Viterbi"
weight:
slug: ""
draft: true # 是否为草稿
comments: true # 本页面是否显示评论
reward: true # 打赏
mermaid: true #是否开启mermaid
showToc: true # 显示目录
TocOpen: true # 自动展开目录
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
showbreadcrumbs: true # 顶部显示路径
cover:
    image: "" #图片路径例如：posts/tech/123/123.png
    zoom: # 图片大小，例如填写 50% 表示原图像的一半大小
    caption: "" #图片底部描述
    alt: ""
    relative: false
---


## Viterbi 알고리즘의 기본 원리
Viterbi 알고리즘은 HMM(Hidden Markov Model)에서 가장 확률이 높은 은닉 상태 시퀀스를 찾기 위한 동적 프로그래밍 알고리즘이다. ASR에서는 관측된 음향 특징(acoustic features)이 주어졌을 때, 가장 확률이 높은 단어나 음소 시퀀스를 찾는 데 사용된다. 


### Viterbi 알고리즘의 주요 구성요소
1. 상태 공간: HMM의 가능한 모든 상태들 ($S = \{s_1, s_2, ..., s_N\}$)
2. 관측 시퀀스: 시간에 따른 음향 특징 벡터 ($O = o_1, o_2, ..., o_T$)
3. 상태 전이 확률: 한 상태에서 다른 상태로 전이할 확률 ($a_{ij}$)
4. 방출 확률: 특정 상태에서 관측값을 생성할 확률 ($b_j(o_t)$)


## Viterbi 알고리즘의 수식

1. 초기화:

$\delta_1(i) = \pi_i \cdot b_i(o_1)$, $1 \leq i \leq N$
$$\psi_1(i) = 0$$
여기서 $\pi_i$는 상태 $i$의 초기 확률, $\delta_t(i)$는 시간 $t$에서 상태 $i$에 도달하는 최대 확률 경로의 확률이다.

2. 재귀:

$$\delta_t(j) = \max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)$$ $$2 \leq t \leq T, 1 \leq j \leq N$$

$$\psi_t(j) = \arg\max_{1 \leq i \leq N} [\delta_{t-1}(i) \cdot a_{ij}]$$

3. 종료:

$$P^* = \max_{1 \leq i \leq N} [\delta_T(i)]$$
$$q_T^* = \arg\max_{1 \leq i \leq N} [\delta_T(i)]$$

4. 경로 역추적:

$$q_t^* = \psi_{t+1}(q_{t+1}^*), t = T-1, T-2, ..., 1$$

## Viterbi의 구체적 예시

무슨 말인지 모르겠다면 예시를 통해 좀 더 쉽게 이해할 수 있다.
음성 인식에서 "나는" 이라는 단어를 인식하는 간단한 예를 통해 Viterbi 알고리즘을 알아보자.

1. 문제 설정:
인식할 단어: "나는"
음소 분해: /ㄴ/, /ㅏ/, /ㄴ/, /ㅡ/, /ㄴ/
각 음소는 3개의 상태를 가진 HMM으로 모델링 (시작, 중간, 끝)
관측 시퀀스: 5개의 음향 특징 벡터 $O = \{o_1, o_2, o_3, o_4, o_5\}$
2. HMM 파라미터:
상태 집합:
/ㄴ1/, /ㄴ2/, /ㄴ3/, /ㅏ1/, /ㅏ2/, /ㅏ3/, /ㄴ1'/, /ㄴ2'/, /ㄴ3'/, /ㅡ1/, /ㅡ2/, /ㅡ3/, /ㄴ1''/, /ㄴ2''/, /ㄴ3''/ (총 15개 상태)

전이 확률 (일부 예시):
$$a_{/ㄴ1/, /ㄴ2/} = 0.7$$ (첫 /ㄴ/의 첫 상태에서 두 번째 상태로)
$$a_{/ㄴ2/, /ㄴ3/} = 0.8$$
$$a_{/ㄴ3/, /ㅏ1/} = 0.9$$ (첫 /ㄴ/의 끝 상태에서 /ㅏ/의 첫 상태로)

방출 확률 (t=1 시점의 예시):
$$b_{/ㄴ1/}(o_1) = 0.4$$
$$b_{/ㅏ1/}(o_1) = 0.1$$
$$b_{/ㅡ1/}(o_1) = 0.05$$

3. Viterbi 알고리즘 실행:

초기화 (t=1):
$$\delta_1(/ㄴ1/) = \pi_{/ㄴ1/} \cdot b_{/ㄴ1/}(o_1) = 1.0 \times 0.4 = 0.4$$
(첫 음소의 첫 상태로 시작한다고 가정)
다른 모든 상태의 초기 확률은 0

t=2 계산:
$$\delta_2(/ㄴ2/) = \delta_1(/ㄴ1/) \cdot a_{/ㄴ1/, /ㄴ2/} \cdot b_{/ㄴ2/}(o_2)$$
$$= 0.4 \times 0.7 \times 0.3 = 0.084$$
$$\delta_2(/ㄴ1/) = \delta_1(/ㄴ1/) \cdot a_{/ㄴ1/, /ㄴ1/} \cdot b_{/ㄴ1/}(o_2)$$
$$= 0.4 \times 0.2 \times 0.25 = 0.02$$

t=3 계산:
$$\delta_3(/ㄴ3/) = \delta_2(/ㄴ2/) \cdot a_{/ㄴ2/, /ㄴ3/} \cdot b_{/ㄴ3/}(o_3)$$
$$= 0.084 \times 0.8 \times 0.5 = 0.0336$$
$$\delta_3(/ㄴ2/) = \max[\delta_2(/ㄴ1/) \cdot a_{/ㄴ1/, /ㄴ2/}, \delta_2(/ㄴ2/) \cdot a_{/ㄴ2/, /ㄴ2/}] \cdot b_{/ㄴ2/}(o_3)$$
$$= \max[0.02 \times 0.7, 0.084 \times 0.1] \times 0.45$$
$$= \max[0.014, 0.0084] \times 0.45 = 0.014 \times 0.45 = 0.0063$$

계속해서 t=4, t=5까지 계산:<br>
마찬가지 방식으로 모든 가능한 상태 전이에 대해 확률을 계산한다.

최종 결과:<br>
가정된 값들로 계산을 완료하면, 가장 높은 확률을 가진 상태 시퀀스:
/ㄴ1/ → /ㄴ2/ → /ㄴ3/ → /ㅏ1/ → /ㅏ2/ → /ㅏ3/ → /ㄴ1'/ → ...


## 방출 확률, 상태 전이 확률
그렇다면 대체 '방출 확률'과 '상태 전이 확률' 이라는 것은 어떻게 구할까?!

### 상태 전이 확률 구하기

1. 전문가 지식 기반 초기화

사용 시점: 모델 학습 시작 전 초기값 설정

- 3-상태 left-to-right HMM 구조에서:
- 자기 루프(self-loop): $a_{ii} \approx 0.6$
- 다음 상태로 전이: $a_{i,i+1} \approx 0.4$
Kaldi의 topo 파일에 이러한 초기값 정의

2. Baum-Welch 알고리즘 (EM 기반)

사용 시점: 모델 학습 과정

- Baum-Welch 알고리즘 (EM 알고리즘의 HMM 버전)
- E(Expectation)-단계: 전방($\alpha$)/후방($\beta$) 확률 계산
- 통계 수집: $\xi_t(i,j)$ (시간 $t$에 상태 $i$, 시간 $t+1$에 상태 $j$에 있을 확률)
- M(Maximization)-단계: 상태 전이 확률 업데이트
$$a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

여기서:<br>
$\xi_t(i,j)$: 시간 $t$에 상태 $i$, 시간 $t+1$에 상태 $j$에 있을 확률<br>
$\gamma_t(i)$: 시간 $t$에 상태 $i$에 있을 확률

3. 강제 정렬(Forced Alignment) 기반

사용 시점: 모델 세련화 및 정제 단계

- 음성-텍스트 쌍이 있는 학습 데이터 준비
- 현재 모델로 발화를 알려진 텍스트와 강제 정렬
- 상태 시퀀스를 카운트하여 전이 확률 계산(카운트를 정규화하여 확률 계산):
$$a_{ij} = \frac{카운트(상태 i에서 j로 전이)}{카운트(상태 i에서의 모든 전이)}$$

### 방출 확률 구하기
GMM은 여러 가우시안 분포의 가중 합으로 표현되는 확률 밀도 함수다. HMM-GMM 시스템에서는 각 HMM 상태의 방출 확률을 GMM으로 모델링한다:<br>
각 HMM 상태 $j$의 방출 확률은 가우시안 혼합 모델로 표현:

$b_j(o_t) = \sum_{m=1}^M c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})$ 방출확률: $b_j(o_t)$
$c_{jm}$: $m$번째 가우시안 컴포넌트의 가중치 (모든 가중치 합은 1)<br>
$\mu_{jm}$: $m$번째 가우시안의 평균 벡터<br>
$\Sigma_{jm}$: $m$번째 가우시안의 공분산 행렬<br>
$\mathcal{N}(o_t; \mu, \Sigma)$: 평균 $\mu$, 공분산 $\Sigma$를 가진 다변량 가우시안 밀도 함수<br>
(나중에 따로 GMM에 대해서 포스팅을 하든지 해야겠다. 이 수식만 봐서는 어떻게 구할 수 있는 것인지 감이 안 잡힌다. 다만 아래 학습 과정으로 이루어 지는 것을 알고 넘어가자)

#### 방출 확률 학습방법
1. GMM 초기화

사용 시점: 모델 학습 시작 전

방법:
- 각 상태에 할당된 특징 벡터의 k-means 클러스터링
- 초기 GMM 컴포넌트 생성 (평균, 공분산, 가중치)
- Kaldi에서는 gmm-init-mono 등의 명령어로 구현

2. Baum-Welch 알고리즘 내 GMM 업데이트

사용 시점: 모델 학습 과정

방법:
- E-단계에서 각 프레임의 상태 소속 확률 $\gamma_t(j)$ 계산
- 각 가우시안 컴포넌트에 대한 책임 확률($\gamma_t(j,m)$) 계산:
$$\gamma_t(j,m) = \gamma_t(j) \cdot \frac{c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})}{\sum_{k=1}^M c_{jk} \mathcal{N}(o_t; \mu_{jk}, \Sigma_{jk})}$$
GMM 파라미터 업데이트:<br>
가중치: $c_{jm} = \frac{\sum_{t=1}^T \gamma_t(j,m)}{\sum_{t=1}^T \gamma_t(j)}$<br>
평균: $\mu_{jm} = \frac{\sum_{t=1}^T \gamma_t(j,m) \cdot o_t}{\sum_{t=1}^T \gamma_t(j,m)}$<br>
공분산: $\Sigma_{jm} = \frac{\sum_{t=1}^T \gamma_t(j,m) \cdot (o_t - \mu_{jm})(o_t - \mu_{jm})^T}{\sum_{t=1}^T \gamma_t(j,m)}$<br>

3. 정렬 기반 GMM 세련화

사용 시점: 모델 세련화 단계

방법:
- 강제 정렬로 특징 벡터를 HMM 상태에 할당
- 정렬된 데이터로 더 복잡한 GMM 학습 (예: 혼합 수 증가)
- Kaldi에서는 gmm-acc-stats-ali와 gmm-est로 구현





#### 책임 확률 수식

$$\gamma_t(j,m) = \gamma_t(j) \cdot \frac{c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})}{\sum_{k=1}^M c_{jk} \mathcal{N}(o_t; \mu_{jk}, \Sigma_{jk})}$$
이 수식은 상태 $j$의 $m$번째 가우시안 컴포넌트가 관측값 $o_t$를 생성할 책임(responsibility) 을 나타낸다. 즉, 관측값 $o_t$가 상태 $j$에서 생성되었다는 조건하에, 그 중에서도 $m$번째 가우시안 컴포넌트에서 생성되었을 확률이다.

#### 방출 확률 수식

$$b_j(o_t) = \sum_{m=1}^M c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})$$
이 수식은 상태 $j$에서 관측값 $o_t$를 생성할 방출 확률(emission probability) 을 나타낸다. GMM으로 모델링된 확률 밀도 함수이다.<br>

#### 두 수식의 관계

중요한 점은 첫 번째 수식의 분모가 바로 두 번째 수식과 같다는 것이다:<br>
$$\sum_{k=1}^M c_{jk} \mathcal{N}(o_t; \mu_{jk}, \Sigma_{jk}) = b_j(o_t)$$
따라서 첫 번째 수식은 다음과 같이 다시 쓸 수 있다:<br>

$$\gamma_t(j,m) = \gamma_t(j) \cdot \frac{c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})}{b_j(o_t)}$$
이는 Baum-Welch 알고리즘의 E-단계에서 계산되는 값으로:<br>
$\gamma_t(j)$: 시간 $t$에 상태 $j$에 있을 확률<br>
$\frac{c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})}{b_j(o_t)}$: 상태 $j$ 내에서 $m$번째 가우시안 컴포넌트의 기여도
두 수식은 다른 의미를 가지지만, GMM 파라미터 학습 과정에서 서로 연관되어 사용된다:

1. 방출 확률 $b_j(o_t)$는 HMM의 기본 구성요소
2. 책임 확률 $\gamma_t(j,m)$은 GMM 파라미터 업데이트에 사용되는 통계량








## DNN-HMM 하이브리드 시스템에서 DNN의 GMM 대체 방식

### 기본 개념의 변화
GMM-HMM 시스템에서 DNN-HMM 하이브리드 시스템으로의 전환은 ASR 발전에 있어 중요한 패러다임 변화였다. 주요 변화는 다음과 같다

GMM-HMM에서
- GMM은 생성 모델(generative model)로 $p(o_t|s_j)$, 즉 상태 $j$가 주어졌을 때 관측값 $o_t$의 우도(likelihood)를 직접 모델링<br>
- 각 HMM 상태마다 별도의 GMM이 존재<br>
- 방출 확률: $b_j(o_t) = p(o_t|s_j) = \sum_{m=1}^M c_{jm} \mathcal{N}(o_t; \mu_{jm}, \Sigma_{jm})$<br>
DNN-HMM에서:<br>
- DNN은 판별 모델(discriminative model)로 $p(s_j|o_t)$, 즉 관측값 $o_t$가 주어졌을 때 상태 $j$의 사후 확률(posterior)을 예측<br>
- 하나의 DNN이 모든 상태의 사후 확률을 동시에 출력<br>
- 베이즈 규칙으로 우도로 변환: $p(o_t|s_j) \propto \frac{p(s_j|o_t)}{p(s_j)}$<br>


### DNN이 GMM을 대체하는 메커니즘
DNN은 다음과 같은 방식으로 GMM의 역할을 대체한다:<br>
입력: 음향 특징(MFCC, FBANK 등)과 그 문맥(앞뒤 프레임)<br>
출력: 각 HMM 상태(senone)에 대한 사후 확률<br>
디코딩 시 사용: 베이즈 규칙을 통해 우도로 변환<br>

#### GMM-HMM으로 초기 강제 정렬 수행
정렬된 프레임 레이블을 사용해 DNN 학습 (교차 엔트로피 손실 함수)<br>
출력층은 softmax 활성화 함수를 통해 모든 HMM 상태에 대한 확률 출력<br>

#### 디코딩 단계
DNN이 각 프레임에 대한 상태 사후 확률 $p(s_j|o_t)$ 출력<br>
사후 확률을 우도로 변환: $p(o_t|s_j) \propto \frac{p(s_j|o_t)}{p(s_j)}$<br>
이 우도를 HMM 디코더에 제공 (Viterbi, Beam Search 등)<br>
여기서 $p(s_j)$는 상태의 사전 확률로, 학습 데이터에서 각 상태의 출현 빈도를 계산하여 얻는다.

### 주요 혁신 포인트

1. 특징 표현력

GMM: 확률 분포를 명시적으로 모델링하지만 복잡한 패턴 인식에 제한적<br>
DNN: 비선형 변환을 통해 더 복잡한 패턴 인식 가능, 더 강력한 특징 표현 학습<br>

2. 문맥 정보 활용

GMM: 주로 현재 프레임의 특징만 사용<br>
DNN: 여러 프레임을 입력으로 받아 더 긴 문맥 정보 활용 가능 (예: 9프레임 윈도우)<br>

3. 파라미터 공유

GMM: 각 상태마다 별도의 파라미터 집합<br>
DNN: 하나의 네트워크로 모든 상태의 확률 계산, 하위 층에서 특징 표현 공유<br>


베이즈 정리를 적용하면:
$P(s_j|o_t) = \frac{P(o_t|s_j) \cdot P(s_j)}{P(o_t)}$



위 식을 $P(o_t|s_j)$에 대해 풀면:
$P(o_t|s_j) = \frac{P(s_j|o_t) \cdot P(o_t)}{P(s_j)}$
이것이 DNN 출력(사후 확률)에서 HMM에 필요한 우도로 변환하는 기본 수식이다.


$P(o_t|s_j) \propto \frac{P(s_j|o_t)}{P(s_j)}$
즉, 스케일링된 우도(scaled likelihood)를 사용한다.
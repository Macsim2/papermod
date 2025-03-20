---
title: "Cross Entropy Loss in Deep Learning"
date: 2025-03-19T01:17:26+09:00
lastmod: 2025-03-19T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- tech
tags: 
- DeepLearning
- Loss Function
- Information Theory
description: "딥러닝에서 Cross Entropy 손실 함수의 이해와 활용"
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

물리학에서는 엔트로피는 시스템의 무질서도 또는 혼란도를 나타내는 척도라고 한다.

열역학 제 2법칙에 따르면 고립된 시스템에서 엔트로피는 시간이 지남에 따라 항상 증가하거나 유지하는 경향을 보인다고 한다. 일상에서는 얼음이 녹거나 폭포가 떨어지는 것도 엔트로피 증가의 예시이다. 조금 어려운 말로는 사용 가능한 에너지에서 사용 불가능한 에너지로 변화하는 경향을 나타내는 개념이다.

그렇다면 deep learning 에서 말하는 cross entropy란 무엇을 말하는걸까?

## Cross Entropy의 기본 개념

Cross Entropy(교차 엔트로피)는 정보 이론에서 유래된 개념으로, 딥러닝에서 가장 널리 사용되는 손실 함수(loss function) 중 하나이다. 특히 분류 문제에서 모델의 예측과 실제 레이블 간의 차이를 측정하는 데 사용된다.

### 직관적 이해

교차 엔트로피는 "두 확률 분포가 얼마나 다른지"를 측정하는 방법이다. 간단히 말하면,
- 실제 정답 분포(target distribution)가 있고
- 모델이 예측한 분포(predicted distribution)가 있을 때
- 이 두 분포가 얼마나 다른지를 수치화한 것이 교차 엔트로피

예를 들어, 고양이/개/새를 분류하는 문제에서:
- 실제 레이블: [1, 0, 0] (이 이미지는 고양이)
- 모델 예측: [0.7, 0.2, 0.1] (모델은 70% 확률로 고양이, 20% 확률로 개, 10% 확률로 새라고 예측)

교차 엔트로피는 이 두 분포 간의 불일치를 수치화하여 모델이 얼마나 정확한지 평가한다.

## 정보 이론적 배경

교차 엔트로피를 이해하기 위해서는 몇 가지 정보 이론 개념을 알아야 한다.

### 엔트로피(Entropy)

엔트로피는 정보의 불확실성을 측정하는 지표이다. 확률 분포 $P$의 엔트로피는 다음과 같이 정의된다:

$$H(P) = -\sum_{i} P(i) \log P(i)$$

여기서:
- $P(i)$는 사건 $i$가 발생할 확률
- 로그의 밑은 보통 $e$(자연로그) 또는 2(비트 단위)를 사용

엔트로피가 높을수록 불확실성이 크고, 낮을수록 확실성이 높다.

### KL 발산(Kullback-Leibler Divergence)

KL 발산은 두 확률 분포 $P$와 $Q$ 사이의 차이를 측정한다:

$$D_{KL}(P||Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

이는 "분포 $P$를 분포 $Q$로 표현할 때 발생하는 추가 정보량"으로 해석할 수 있다.

### 교차 엔트로피

교차 엔트로피는 KL 발산과 밀접하게 관련되어 있다:

$$H(P, Q) = H(P) + D_{KL}(P||Q) = -\sum_{i} P(i) \log Q(i)$$

여기서:
- $H(P)$는 분포 $P$의 엔트로피
- $H(P, Q)$는 분포 $P$와 $Q$ 사이의 교차 엔트로피

분류 문제에서 실제 레이블 분포 $P$는 고정되어 있으므로, 교차 엔트로피를 최소화하는 것은 KL 발산을 최소화하는 것과 동일하다.

## 딥러닝에서의 Cross Entropy 손실 함수

딥러닝 모델, 특히 분류 모델을 학습시킬 때 교차 엔트로피 손실 함수가 널리 사용되는 이유를 알아보자.

### Binary Cross Entropy (BCE)

이진 분류 문제(두 클래스 분류)에서 사용되는 손실 함수이다:

$$BCE = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

여기서:
- $N$은 샘플 수
- $y_i$는 실제 레이블(0 또는 1)
- $p_i$는 모델이 예측한 양성 클래스의 확률

### Categorical Cross Entropy (CCE)

다중 클래스 분류 문제에서 사용되는 손실 함수이다:

$$CCE = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C}y_{ij} \log(p_{ij})$$

여기서:
- $N$은 샘플 수
- $C$는 클래스 수
- $y_{ij}$는 샘플 $i$가 클래스 $j$에 속하는지 여부(0 또는 1)
- $p_{ij}$는 모델이 예측한 샘플 $i$가 클래스 $j$에 속할 확률

### Sparse Categorical Cross Entropy

범주형 교차 엔트로피와 동일하지만, 원-핫 인코딩된 벡터 대신 정수 형태의 레이블을 사용한다. 클래스 수가 많을 때 메모리 효율성이 좋다.

## Cross Entropy의 장점

딥러닝에서 교차 엔트로피가 손실 함수로 널리 사용되는 이유는 다음과 같다:

### 1. 확률적 해석이 명확함

교차 엔트로피는 확률 분포 간의 차이를 직접 측정하기 때문에, 분류 문제에서 자연스러운 선택이다.

### 2. 그래디언트 소실 문제를 완화

시그모이드나 소프트맥스 활성화 함수와 함께 사용할 때, 평균 제곱 오차(MSE)에 비해 그래디언트 소실 문제가 덜 심각하다.

| 손실 함수 | 그래디언트 |
|---------|----------|
| MSE + 시그모이드 | $\frac{\partial L}{\partial z} = (a-y) \cdot a \cdot (1-a)$ |
| 교차 엔트로피 + 시그모이드 | $\frac{\partial L}{\partial z} = a-y$ |

교차 엔트로피의 경우 그래디언트가 출력 활성화($a$)와 타겟($y$)의 차이에 직접 비례하므로, 활성화 함수의 도함수가 포함되지 않아 그래디언트 소실이 덜 발생한다.

### 3. 최대 우도 추정(MLE)와의 연관성

교차 엔트로피 최소화는 통계학에서 최대 우도 추정과 수학적으로 동일하다. 따라서 확률론적 관점에서 이론적으로 견고한 기반을 가진다.

### 4. 분류 경계에 집중

교차 엔트로피는 확신이 낮은 예측(확률이 0.5에 가까운)에 더 큰 페널티를 부과한다. 이는 분류 경계 부근의 샘플을 더 잘 구분하도록 모델을 학습시킨다.

## Binary Cross Entropy vs Categorical Cross Entropy

| 특성 | Binary Cross Entropy | Categorical Cross Entropy |
|-----|---------------------|--------------------------|
| 적용 분야 | 이진 분류 (2개 클래스) | 다중 클래스 분류 (3개 이상) |
| 출력 활성화 함수 | 시그모이드 | 소프트맥스 |
| 출력 형태 | 단일 값 (0~1 사이) | 각 클래스별 확률 벡터 |
| 타겟 형태 | 단일 값 (0 또는 1) | 원-핫 인코딩 벡터 |
| 다중 레이블 분류 가능 여부 | 가능 (각 출력을 독립적으로 처리) | 불가능 (소프트맥스는 상호 배타적) |

## 실제 구현 및 사용 예시

### PyTorch 예시

```python
import torch
import torch.nn as nn

# 이진 분류를 위한 BCE
criterion = nn.BCELoss()
outputs = torch.tensor([0.7, 0.3, 0.9])
targets = torch.tensor([1.0, 0.0, 1.0])
loss = criterion(outputs, targets)
print(f"Binary Cross Entropy Loss: {loss.item()}")

# 다중 클래스 분류를 위한 CCE
criterion = nn.CrossEntropyLoss()
outputs = torch.tensor([[0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
targets = torch.tensor([1, 0])  # 클래스 인덱스
loss = criterion(outputs, targets)
print(f"Categorical Cross Entropy Loss: {loss.item()}")
```

### TensorFlow/Keras 예시

```python
import tensorflow as tf
from tensorflow import keras

# 이진 분류 모델
model = keras.Sequential([
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 다중 클래스 분류 모델
model = keras.Sequential([
    keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # 원-핫 인코딩된 타겟
    # 또는 loss='sparse_categorical_crossentropy',  # 정수 타겟
    metrics=['accuracy']
)
```

## 주의사항 및 실용적 팁

### 1. 수치적 안정성

로그 함수는 입력이 0에 가까울 때 불안정해진다. 따라서 구현 시 작은 epsilon 값을 더해 안정성을 확보하는 것이 중요하다:

```python
def stable_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))
```

### 2. 클래스 불균형 문제

클래스 분포가 불균형한 데이터셋에서는 가중치를 적용한 교차 엔트로피를 사용할 수 있다:

```python
# PyTorch에서 가중치 적용
weight = torch.tensor([1.0, 5.0, 2.0])  # 클래스별 가중치
criterion = nn.CrossEntropyLoss(weight=weight)

# TensorFlow/Keras에서 가중치 적용
model.compile(
    loss=keras.losses.CategoricalCrossentropy(
        class_weights={0: 1.0, 1: 5.0, 2: 2.0}
    )
)
```

### 3. 레이블 스무딩(Label Smoothing)

모델의 과신(overconfidence)을 방지하고 일반화 성능을 향상시키기 위해 레이블 스무딩을 적용할 수 있다:

```python
# PyTorch에서 레이블 스무딩
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# TensorFlow/Keras에서 레이블 스무딩
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
)
```

레이블 스무딩은 원-핫 인코딩된 타겟을 약간 "부드럽게" 만들어, 1은 0.9로, 0은 0.1/(클래스 수-1)로 변환한다.

## 결론

교차 엔트로피는 분류 문제에서 이론적으로 견고하고 실용적인 손실 함수이다. 정보 이론에 기반한 확률적 해석이 명확하고, 그래디언트 계산이 효율적이며, 실제로 우수한 성능을 발휘한다. 

모델의 예측을 확률 분포로 해석할 수 있는 거의 모든 분류 문제에서 기본적으로 선택할 수 있는 손실 함수이며, 이진 분류와 다중 클래스 분류 각각에 맞는 변형을 사용할 수 있다.

딥러닝에서 교차 엔트로피의 중요성은 아무리 강조해도 지나치지 않으며, 모델의 학습과 성능을 좌우하는 핵심 요소 중 하나이다.

## 참고 문헌

1. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. 
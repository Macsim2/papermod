---
title: "Beam Search"
date: 2025-03-15T01:17:26+09:00
lastmod: 2025-03-15T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- tech
tags: 
- ASR
- beamsearch
description: "about Beam Search"
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



## Beam Search의 기본 개념과 원리
beam search, 음성 인식을 구현 해본 사람이라면, 한번 쯤은 들어보게 되는 용어이다. 다른 분야에서는 많이 사용하는지는 모르겠지만 search 알고리즘의 갈래이기도 하고, 선택지가 많아지는 task인 경우 사용할 수 있는 유용한(?) 방법 같다. 그래서 오늘은 ASR을 모델링 해본 사람으로써 beam search에 대해서 글로써 정리해보고자 한다. 주로 코드로 접하는 beam search 이지만 이번 기회에 개념부터 알아보자.


### Beam Search란 무엇인가?

Beam Search는 자동 음성 인식(ASR) 시스템에서 디코딩 과정에 사용되는 핵심 알고리즘이다. 음성을 텍스트로 변환할 때, 입력된 음향 신호에 대해 가능한 여러 가설(hypothesis)들을 효율적으로 탐색하여 가장 확률이 높은 시퀀스를 찾아내는 방법이다.<br>
ASR에서 디코딩은 다음과 같은 수식으로 표현할 수 있다:

$$Y^* = \arg\max_Y P(Y|X)$$

이 수식이 ASR task를 한마디로 하자면, 최종 수식이다. 여기서 $X$는 입력 음향 특징(acoustic feature), $Y$는 출력 텍스트 시퀀스를 의미한다. 이론적으로는 모든 가능한 $Y$에 대해 확률을 계산하여 최대값을 찾아야 하지만, 실제로는 가능한 경우의 수가 너무 많아 전수 조사(exhaustive search)가 불가능하다. 그리고 실제로 전수 조사를 하는 컴퓨팅 비용대비 얻는 보상이 크지 않다. 가성비가 떨어지는 느낌이랄까?

### Greedy Search와의 차이점

가장 단순한 디코딩 방법은 Greedy Search(탐욕 탐색)이다. 이 방법은 각 시점마다 가장 확률이 높은 하나의 토큰만을 선택하여 시퀀스를 구성한다.<br>
$$y_t = \arg\max_{y} P(y|X, y_1, y_2, \ldots, y_{t-1})$$<br>
그러나 Greedy Search는 지역 최적해(local optimum)에 쉽게 빠질 수 있으며, 전체적으로 최적의 시퀀스를 놓칠 가능성이 크다. 예를 들어, 특정 시점에서 두 번째로 확률이 높은 선택이 이후의 단계에서 더 좋은 전체 경로로 이어질 수 있는데, Greedy Search는 이러한 가능성을 고려하지 않는다.
반면 Beam Search는 각 디코딩 단계에서 상위 k개의 가능성(beam)을 유지하며 탐색한다. 이를 통해 지역 최적해에 빠질 위험을 줄이고, 전체적으로 더 나은 해를 찾을 확률을 높인다.

### Beam Search의 작동 원리

Beam Search의 기본 알고리즘은 다음과 같다:
1. 초기 상태에서 시작하여 첫 번째 타임 스텝에서 가능한 모든 토큰의 확률을 계산한다.<br>
2. 확률이 가장 높은 상위 k개의 토큰을 선택한다 (k는 beam width 또는 beam size).<br>
3. 선택된 각 토큰에 대해, 다음 타임 스텝에서 가능한 모든 토큰의 확률을 계산한다.<br>
4. 이전 경로의 확률과 현재 토큰의 확률을 결합하여 $k \times |V|$ 개의 가능한 경로 중 상위 k개를 다시 선택한다 (|V|는 어휘 크기).<br>
5. 종료 토큰이 나타나거나 최대 길이에 도달할 때까지 3-4 단계를 반복한다.<br>

수식적으로 표현하면, 시간 $t$에서의 경로 점수는 다음과 같이 계산된다:
$$score(y_1, y_2, \ldots, y_t) = \log P(y_1, y_2, \ldots, y_t|X) = \sum_{i=1}^{t} \log P(y_i|X, y_1, \ldots, y_{i-1})$$

4번 설명에서 '이전 경로의 확률과 현재 토큰의 확률의 결합'은 단순 multiply다. 그러나 확률은 0과 1사이의 값이므로 타입 스텝 $t$가 길어지면 0보다 작은 확률값 $y_t$가 점점 0과 가까워 질 것이므로 컴퓨터 연산에서 underflow 문제를 야기할 수 있다. 그래서 로그 확률의 합으로, 수치적 안정성을 위해 일반적으로 로그 도메인에서 계산함으로 문제를 해결한다.

### Beam Width(Size)의 의미와 선택

Beam width(k)는 각 디코딩 단계에서 유지하는 가설의 수를 결정하는 매개변수이다. 이는 탐색 공간과 계산 복잡성, 결과의 품질 사이의 중요한 균형점을 제공한다.<br>
작은 beam width (k=1):<br> Greedy Search와 동일하며, 계산이 빠르지만 최적해를 놓칠 가능성이 높다.<br>
큰 beam width:<br> 더 넓은, 종합적인 탐색을 가능하게 하지만 계산 비용이 증가한다.
ASR 시스템에서는 일반적으로 k=5~20 사이의 값을 사용하는데, 이는 디코딩 품질과 속도 사이의 합리적인 타협점을 제공한다. 그러나 최적의 beam width는 모델 아키텍처, 태스크 복잡성, 하드웨어 리소스 등에 따라 달라질 수 있다.

흥미롭게도, beam width를 무한히 크게 하는 것이 항상 최상의 결과를 보장하지는 않는다고 한다. 이는 'beam search curse'라는 현상으로, 특정 임계값 이상에서는 성능 향상이 미미하거나 오히려 감소할 수 있다. 그렇다고 한다.


## ASR 모델 아키텍처별 Beam Search 적용

### HMM-GMM에서의 Beam Search
HMM-GMM 시스템에서 Beam Search는 다음과 같이 작동한다:
각 시간 프레임마다 가능한 모든 HMM 상태의 확률을 계산한다.
확률이 임계값(beam threshold) 이하인 상태는 제거(pruning)한다.
남은 활성 상태(active states)에 대해서만 다음 프레임의 계산을 진행한다.<br>
수식적으로, 시간 $t$에서 상태 $s$의 Viterbi 경로 확률은:
$$\delta_t(s) = \max_{s'} \{\delta_{t-1}(s') \cdot a_{s's} \cdot b_s(o_t)\}$$

여기서:

$\delta_t(s)$: 시간 $t$에서 상태 $s$까지의 최대 경로 확률<br>
$a_{s's}$: 상태 $s'$에서 $s$로의 전이 확률<br>
$b_s(o_t)$: 상태 $s$에서 관측값 $o_t$의 방출 확률<br>

Beam Search에서는 각 시간 프레임마다 다음 조건을 만족하는 상태만 유지한다:
$$\delta_t(s) \geq \max_j \{\delta_t(j)\} \cdot \theta$$
여기서 $\theta$는 beam threshold(0과 1 사이의 값)이다.
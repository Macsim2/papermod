---
title: "FourierTransform"
date: 2025-03-02T01:17:26+09:00
lastmod: 2025-03-02T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- 
tags: 
- 
description: "about FT"
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

< Index >

* Fourier Transform
    *   [Fourier Series and Periodic Signals](#fourier-series-and-periodic-signals)
    *   [Mathematical Definition of Continuous Fourier Transform (CFT)](#mathematical-definition-of-continuous-fourier-transform-cft)
    *   [Discrete Fourier Transform (DFT) and Real-World Applications](#discrete-fourier-transform-dft-and-real-world-applications)
    *   [Fast Fourier Transform (FFT): Concept and Efficiency](#fast-fourier-transform-fft-concept-and-efficiency)



## Fourier Series and Periodic Signals

주기 신호란 일정 시간마다 동일한 패턴이 반복되는 신호를 말한다. 주기 $T$를 갖는 함수 $f(t)$는 다음 조건을 만족한다.
$$
f(t) = f(t + T) \quad \text{모든 } t \text{에 대해}
$$
푸리에는 이러한 주기 함수가 다음과 같이 무한개의 사인과 코사인의 합으로 표현될 수 있음을 증명했다.
$$
f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right) \right]
$$

여기서:
$a_0, a_n, b_n$은 푸리에 계수로, 원래 함수와 사인/코사인 함수 간의 상관관계를 나타낸다.<br>
$n$은 각 주파수 성분의 고조파(harmonic) 차수를 나타낸다.<br>
$T$는 신호의 주기다.

푸리에 계수는 다음과 같이 계산한다.
$$
a_0 = \frac{2}{T} \int_{t_0}^{t_0+T} f(t) \, dt
$$
$$
a_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(t) \cos\left(\frac{2\pi n t}{T}\right) \, dt \quad \text{for } n \geq 1
$$
$$
b_n = \frac{2}{T} \int_{t_0}^{t_0+T} f(t) \sin\left(\frac{2\pi n t}{T}\right) \, dt \quad \text{for } n \geq 1
$$
이 계수들은 각 주파수 성분의 '강도'를 나타낸다. 예를 들어, 440Hz의 순수한 음은 440Hz 성분의 계수만 크고 나머지는 0에 가깝겠지만, 복잡한 악기 소리는 다양한 주파수에 여러 크기의 계수들이 존재한다.
복소수 형태로는 오일러 공식($e^{ix} = \cos x + i \sin x$)을 이용해 더 간결하게 표현할 수 있다.
$$
f(t) = \sum_{n=-\infty}^{\infty} c_n e^{i \frac{2\pi n t}{T}}
$$
여기서 푸리에 계수 $c_n$은:
$$
c_n = \frac{1}{T} \int_{t_0}^{t_0+T} f(t) e^{-i \frac{2\pi n t}{T}} \, dt
$$
[그림 필요: 사각파, 삼각파 등 간단한 주기 신호와 이를 구성하는 다양한 주파수의 사인파들을 보여주는 그림. 성분의 개수가 증가할수록 원래 신호에 가까워지는 과정을 시각화]


<!-- ## <a id="mathematical-definition-of-continuous-fourier-transform-cft"></a> Mathematical Definition of Continuous Fourier Transform (CFT) -->
## Mathematical Definition of Continuous Fourier Transform (CFT)

푸리에 급수는 주기적인 신호만 다룰 수 있다는 한계가 있다. 실제 오디오 신호와 같은 비주기적 신호를 분석하기 위해서는 연속 푸리에 변환(CFT)이 필요하다.
연속 푸리에 변환은 주기가 무한대인 신호, 즉 비주기 신호를 처리할 수 있도록 푸리에 급수를 확장한 개념이다. 주기가 무한대로 늘어나면 주파수 성분들 사이의 간격은 무한히 작아지고, 결국 연속적인 주파수 스펙트럼이 된다.
함수 $f(t)$의 연속 푸리에 변환 $F(\omega)$는 다음과 같이 정의한다:
$$
F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} \, dt
$$
그리고 역변환은:
$$
f(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} F(\omega) e^{i\omega t} \, d\omega
$$
여기서:
$\omega = 2\pi f$는 각주파수(angular frequency)를 나타낸다.
$F(\omega)$는 $f(t)$의 주파수 스펙트럼을 나타내며, 각 주파수 성분의 복소 진폭을 제공한다.<br>
진폭의 절대값 $|F(\omega)|$는 각 주파수 성분의 강도(magnitude spectrum)를 나타낸다.<br>
위상 $\angle F(\omega)$는 각 주파수 성분의 시간 지연(phase spectrum)을 나타낸다.<br>
오디오 신호 처리에서는 주로 주파수 $f$(Hz 단위)를 사용하므로, 다음과 같이 표기할 수 있다:
$$
F(f) = \int_{-\infty}^{\infty} f(t) e^{-i 2\pi f t} \, dt
$$
연속 푸리에 변환을 통해 어떤 복잡한 소리든 다양한 주파수 성분으로 분해할 수 있으며, 이는 소리의 특성을 이해하고 분석하는 기본 도구이다.<br>
[그림 필요: 음성 신호나 음악의 한 부분에 대한 시간 도메인 파형과 해당 신호의 연속 푸리에 변환 결과인 주파수 스펙트럼을 시각화한 그림]



## Discrete Fourier Transform (DFT) and Real-World Applications


실제 오디오 처리에서는 연속 신호가 아닌 이산(discrete) 신호를 다룬다. 오디오를 디지털로 변환하면 연속적인 파형이 일정 간격으로 샘플링된 이산 값들의 시퀀스가 된다. 이러한 이산 신호에 적용되는 것이 이산 푸리에 변환(DFT)이다.
길이가 $N$인 이산 시퀀스 $x[n]$ ($n = 0, 1, 2, ..., N-1$)의 DFT는 다음과 같이 정의하게 된다:
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i\frac{2\pi}{N}kn} \quad \text{for } k = 0, 1, 2, ..., N-1
$$
그리고 역변환은:
$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i\frac{2\pi}{N}kn} \quad \text{for } n = 0, 1, 2, ..., N-1
$$
여기서:

$X[k]$는 DFT의 출력에서 $k$번째 주파수 성분(빈, bin)을 의미하며, 이는 신호의 주파수 도메인 표현에서 특정 주파수 성분의 복소수 값을 나타낸다. 실수부는 해당 주파수에서의 코사인 성분, 허수부는 사인 성분을 나타낸다.<br>
$k$는 주파수 인덱스로, 실제 주파수는 $f_k = k \cdot f_s / N$으로 계산되고, 여기서 $f_s$는 샘플링 주파수이다.<br>
좀 더 이해를 하기 위해 직접적인 값을 이용하자면, $f_s = 16,000\text{Hz}$, $N = 512$일 때, $k = 1$의 빈은 주파수 $f_1 = \frac{1 \times 16,000}{512} = 31.25\text{Hz}$를 나타낸다. 이는 스펙트로그램에서 주파수 축의 한 점에 해당한다.  <br>
512 샘플 프레임에서 DFT를 계산하면, k=0부터 k=511까지 총 512개의 빈이 생성된다. 여기서 k=0은 0Hz(직류 성분)를 나타내고, k=1부터 k=256까지는 0에서 8,000Hz까지의 양의 주파수를 커버한다. 특히, k=256은 샘플링 주파수의 절반인 8,000Hz(나이퀴스트 주파수)에 해당한다. <br>

k=257부터 k=511까지는 음의 주파수에 해당하며, 실수 신호의 경우 이 부분은 k=0부터 k=255까지의 대칭적인 복소 켤레 성분으로 해석된다. 따라서 실제 분석에서는 보통 k=0부터 k=256까지(257개 빈)를 주로 사용하게 된다.<br>

또 하나 알아둬야 할 점이 있다. 비로 시간과 주파수 해상도의 트레이드오프 이다.
시간 해상도와 주파수 해상도는 상호 보완 적이다. 작은 N(예: 256 샘플)은 각 프레임이 짧아져 빠른 시간 변화(예: 음성의 자음)를 더 잘 포착할 수 있다.(시간 해상도 향상). 하지만 주파수 해상도는 낮아져, 비슷한 주파수 성분을 구분하기 어려워진다.(주파수 간격이 넓어짐).

반대로, 큰 N(예: 1,024 샘플)은 더 긴 프레임을 분석하므로 주파수 해상도가 높아져(주파수 간격이 좁아짐) 소리의 저주파 성분을 정밀하게 분석할 수 있지만, 시간 해상도가 낮아져 빠른 변화 포착이 어려워진다.

그리고 DFT는 이산 신호에 대한 정확한 주파수 분석을 제공하지만, 계산 복잡도가 $O(N^2)$로 높다는 단점이 있다. 이는 다음에서 설명할 고속 푸리에 변환(FFT)으로 해결할 수 있다.<br>

[그림 필요: 디지털 오디오 신호의 샘플링 과정과 DFT 적용 전후의 비교, 그리고 샘플 개수에 따른 주파수 해상도 변화를 시각화한 그림]


## Fast Fourier Transform (FFT): Concept and Efficiency

고속 푸리에 변환(FFT)은 DFT를 효율적으로 계산하기 위한 알고리즘이다. 1960년대 James Cooley와 John Tukey가 개발한 이 알고리즘은 DFT의 계산 복잡도를 $O(N^2)$에서 $O(N \log N)$으로 크게 줄였다<br>
FFT의 핵심 아이디어는 '분할 정복(divide and conquer)' 방식으로, 큰 DFT 계산을 더 작은 DFT들로 재귀적으로 분해하는 것이다. 가장 널리 사용되는 FFT 알고리즘은 Cooley-Tukey 알고리즘으로, 다음과 같은 원리로 작동한다:<br>
1. $N$-포인트 DFT를 짝수 인덱스와 홀수 인덱스로 분리한다.
2. 짝수 인덱스의 $N/2$-포인트 DFT와 홀수 인덱스의 $N/2$-포인트 DFT를 계산한다.
3. 이 두 결과를 조합하여 원래의 $N$-포인트 DFT를 얻는다.
<br>
이를 수식으로 표현하면:
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i\frac{2\pi}{N}kn} = \sum_{m=0}^{N/2-1} x[2m] e^{-i\frac{2\pi}{N}k(2m)} + \sum_{m=0}^{N/2-1} x[2m+1] e^{-i\frac{2\pi}{N}k(2m+1)}
$$
이는 다음과 같이 다시 작성할 수 있다:
$$
X[k] = E[k] + e^{-i\frac{2\pi}{N}k} O[k] \quad \text{for } k = 0, 1, 2, ..., N-1
$$
여기서:
$E[k]$는 짝수 인덱스 샘플의 $N/2$-포인트 DFT<br>
$O[k]$는 홀수 인덱스 샘플의 $N/2$-포인트 DFT<br>
예를들면 $X[8]$ 을 구하기 위해 $X[4]$ 과 $X[3]$ 을 가지고 구할 수 있다는 얘기이다.<br>
FFT는 시퀀스가 주기적이라고 가정하며, 비주기적 신호에서는 스펙트럼 누출이 발생할 수 있다 그래서 이를 완화하기 위해 핸(Hann), 해밍(Hamming) 등의 창 함수를 사용한다고 한다.<br>
FFT 알고리즘은 데이터 크기가 2의 거듭제곱일 때 가장 효율적이지만, 다른 크기에 대해서도 최적화된 알고리즘이 개발되어 있다.<br>


[그림 필요: FFT의 분할 정복 알고리즘을 단계별로 시각화한 그림, 그리고 DFT와 FFT의 계산 복잡도 차이를 보여주는 그래프]
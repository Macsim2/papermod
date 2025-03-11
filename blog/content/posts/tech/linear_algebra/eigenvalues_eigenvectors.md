---
title: "Eigenvalues & Eigenvectors"
date: 2025-03-03T01:17:26+09:00
lastmod: 2025-03-03T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- tech
tags: 
- linear_algebra
description: "about Eigenvalues & Eigenvectors"
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


* Eigenvalues & Eigenvectors
    * [기본 개념](#기본-개념)
    * [수학적 정의](#수학적-정의)
    * [기하학적 의미](#기하학적-의미)
    * [주요 성질](#주요-성질)
    * [대각화](#대각화diagonalization)
    * [고유값 분해](#고유값-분해eigendecomposition)
    * [응용 분야](#응용-분야)
    * [고유값 계산 방법](#고유값-계산-방법)
    * [특이값 분해와의 관계](#특이값-분해svd와의-관계)

 => 실제 응용: 구글의 PageRank 알고리즘, 주성분 분석(PCA), 진동 모드 분석

## 기본 개념

고유값과 고유벡터는 선형대수학에서 가장 중요한 개념 중 하나이다. 정방행렬 $A$에 대하여, 0이 아닌 벡터 $v$와 스칼라 $\lambda$가 다음 방정식을 만족할 때:

$$Av = \lambda v$$

$\lambda$를 행렬 $A$의 **고유값(eigen value)** 이라 하고, $v$를 $\lambda$에 대응하는 **고유벡터(eigen vector)** 라고 한다. <br>
직관적으로 이해 하자면 고유벡터는 행렬 $A$에 의한 선형변환이 적용될 때 방향이 변하지 않고 오직 크기만 $\lambda$배 변하는 특별한 벡터이다. <br>
즉, 행렬 $A$가 고유벡터 $v$에 작용하면 $v$의 방향은 그대로 유지된다.<br>
또, 고유벡터의 크기는 고유값 $\lambda$에 비례하여 늘어나거나 줄어든다.<br>

## 수학적 정의<br>
- **고유방정식**: $Av = \lambda v$<br>
- **특성방정식**: $\det(A - \lambda I) = 0$<br>
- **고유공간**: 고유값 $\lambda$에 대응하는 모든 고유벡터의 집합 $E_{\lambda} = \{v \neq 0 : Av = \lambda v\}$<br>

예를 들어, 행렬 $A = \begin{bmatrix} 3 & 1 \\\\ 1 & 3 \end{bmatrix}$의 경우<br>
특성방정식: $\det(A - \lambda I) = \det\begin{bmatrix} 3-\lambda & 1 \\\\ 1 & 3-\lambda \end{bmatrix} = (3-\lambda)^2 - 1 = 0$
이를 풀면 $\lambda = 2$ 또는 $\lambda = 4$가 된다.
$\lambda = 2$일 때의 고유벡터는 $v_1 = \begin{bmatrix} -1 \\\\ 1 \end{bmatrix}$ (또는 이의 스칼라 배)
$\lambda = 4$일 때의 고유벡터는 $v_2 = \begin{bmatrix} 1 \\\\ 1 \end{bmatrix}$ (또는 이의 스칼라 배)

행렬식(determinant, $\det$)은 정방행렬에 대해 정의되는 스칼라 값이다.



## 기하학적 의미

고유벡터는 선형변환 $A$에 의해 방향이 변하지 않고 오직 크기만 $\lambda$배 변하는 특별한 벡터이다. 이것은 다음과 같은 의미를 갖는다.

- $\lambda > 0$: 고유벡터는 같은 방향으로 늘어나거나 줄어든다
- $\lambda < 0$: 고유벡터는 반대 방향으로 늘어나거나 줄어든다
- $|\lambda| = 1$: 고유벡터의 길이가 보존된다
- $\lambda = 0$: 고유벡터는 영벡터로 매핑된다

## 주요 성질

1. $n \times n$ 행렬은 최대 $n$개의 서로 다른 고유값을 가진다.
2. 대칭행렬($A = A^T$)의 모든 고유값은 실수이다.
3. 직교행렬($A^TA = I$)의 모든 고유값의 절댓값은 1이다.
4. 행렬 $A$의 대각합(trace)은 모든 고유값의 합과 같다: $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$
5. 행렬 $A$의 행렬식은 모든 고유값의 곱과 같다: $\det(A) = \prod_{i=1}^{n} \lambda_i$

## 대각화(Diagonalization)

대각화는 주어진 정방행렬을 유사한 대각행렬로 변환하는 과정이다.<br>
쉽게 말해, 
#### 복잡한 행렬을 대각 원소만 값을 가지고 나머지는 모두 0인 더 단순한 형태의 행렬로 바꾸는 것 이다.

## 고유값 분해(Eigen decomposition)

#### 고유값 분해는 정방행렬을 고유값과 고유벡터를 이용하여 대각화하는 방법이다.
이는 복잡한 행렬을 더 단순한 형태로 표현하여 계산과 분석을 용이하게 한다.<br>
대각화 가능한 $n \times n$ 행렬 $A$는 다음과 같이 분해할 수 있다.

$$A = PDP^{-1}$$

여기서:
- $P$는 $A$의 고유벡터들을 열로 갖는 행렬이다
- $D$는 대응하는 고유값들을 대각선에 갖는 대각행렬이다



대각행렬 $D$는 다음과 같은 형태를 가진다.
$$D = \begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\\\
0 & \lambda_2 & \cdots & 0 \\\\
\vdots & \vdots & \ddots & \vdots \\\\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}$$
대각 원소 $\lambda_1, \lambda_2, \ldots, \lambda_n$은 원래 행렬 $A$의 고유값들이다.

위의 예제를 가지고 대각행렬 $D$ 구해보자.<br>
행렬 $A = \begin{bmatrix} 3 & 1 \\\\ 1 & 3 \end{bmatrix}$<br>
고유값: $\lambda_1 = 4$, $\lambda_2 = 2$<br>
고유벡터: $v_1 = \begin{bmatrix} 1 \\\\ 1 \end{bmatrix}$, $v_2 = \begin{bmatrix} 1 \\\\ -1 \end{bmatrix}$<br>
행렬 $P = \begin{bmatrix} 1 & 1 \\\\ 1 & -1 \end{bmatrix}$<br>
대각행렬 $D = \begin{bmatrix} 4 & 0 \\\\ 0 & 2 \end{bmatrix}$<br>
$PDP^{-1} = A$ 임을 검증을 마지막으로 대각행렬 $D$를 확인한다. 이로써 행렬 $A$ 대신 더 단순한 대각행렬 $D$로 계산을 수행할 수 있다.

그렇다면 $3X3$ 이상의 행렬에 대해서는 어떻게 고유값과 고유벡터를 구할 수 있을까?<br>
1. 특성다항식 설정: $\det(A - \lambda I) = 0$
2. 행렬식 계산: 3×3 이상 행렬의 행렬식은 다음과 같은 방법으로 계산한다.
* 여인수 전개(cofactor expansion)
* 행 연산을 통한 상삼각/하삼각 행렬로의 변환
<br>
3. 다항식 근 구하기: 특성다항식의 근이 행렬의 고유값이다.

그런 다음, 이제 고유벡터를 구해주면 된다.
1. 연립방정식 설정: $(A - \lambda I)v = 0$
2. 기약행사다리꼴(RREF)로 변환: 가우스-조던 소거법 적용
3. 해공간(null space) 구하기: 기저 벡터들이 고유벡터이다.

<br>

실제 행렬이 큰 경우 위 방법이 어렵거나 비효율적일 수 있어 다음의 방법을 사용한다고 한다.<br>

1. **QR 알고리즘**: 큰 행렬의 고유값을 수치적으로 계산하는 데 효율적이다
2. **멱승법(Power method)**: 가장 큰 절댓값을 갖는 고유값과 그에 대응하는 고유벡터를 찾는다
3. **역멱승법(Inverse power method)**: 특정 값 근처의 고유값을 찾는 데 사용된다


행렬 $A = \begin{bmatrix} 1 & 2 & 0 \\\\ 0 & 3 & 0 \\\\ 2 & -4 & 2 \end{bmatrix}$의 고유값과 고유벡터를 구해보자.<br>
고유값 구하기:
특성다항식:
$$\det(A - \lambda I) = \det\begin{bmatrix} 1-\lambda & 2 & 0 \\\\ 0 & 3-\lambda & 0 \\\\ 2 & -4 & 2-\lambda \end{bmatrix} = 0$$
행렬식 계산(첫 번째 열 기준 여인수 전개):
$$\begin{align}
&(1-\lambda)\det\begin{bmatrix} 3-\lambda & 0 \\\\ -4 & 2-\lambda \end{bmatrix} - 0 + 2\det\begin{bmatrix} 2 & 0 \\\\ 3-\lambda & 0 \end{bmatrix}\\\\
&= (1-\lambda)[(3-\lambda)(2-\lambda) - 0] + 2[0]\\\\
&= (1-\lambda)(3-\lambda)(2-\lambda)
\end{align}$$
고유값: $\lambda_1 = 1$, $\lambda_2 = 3$, $\lambda_3 = 2$ <br>
고유벡터 구하기:
$\lambda_1 = 1$일 때:<br>
$(A - I)v = 0$ 풀기
$\begin{bmatrix} 0 & 2 & 0 \\\\ 0 & 2 & 0 \\\\ 2 & -4 & 1 \end{bmatrix}\begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \end{bmatrix}$
해공간 구하기: $v_1 = \begin{bmatrix} -1 \\\\ 0 \\\\ \frac{1}{2} \end{bmatrix}$ 또는 이의 스칼라 배<br><br>
$\lambda_2 = 3$일 때:<br>
$(A - 3I)v = 0$ 풀기
$\begin{bmatrix} -2 & 2 & 0 \\\\ 0 & 0 & 0 \\\\ 2 & -4 & -1 \end{bmatrix}\begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \end{bmatrix}$
해공간 구하기: $v_2 = \begin{bmatrix} 1 \\\\ 1 \\\\ 0 \end{bmatrix}$ 또는 이의 스칼라 배<br><br>
$\lambda_3 = 2$일 때:<br>
$(A - 2I)v = 0$ 풀기
$\begin{bmatrix} -1 & 2 & 0 \\\\ 0 & 1 & 0 \\\\ 2 & -4 & 0 \end{bmatrix}\begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \end{bmatrix} = \begin{bmatrix} 0 \\\\ 0 \\\\ 0 \end{bmatrix}$
해공간 구하기: $v_3 = \begin{bmatrix} 0 \\\\ 0 \\\\ 1 \end{bmatrix}$ 또는 이의 스칼라 배<br><br>

위 연립 방정식은 가우스-조던 소거법을 적용하여 구하는 것이 가능하다.<br>


## 특이값 분해(SVD)와의 관계

특이값 분해는 고유값 분해의 일반화로, 정방행렬이 아닌 행렬에도 적용할 수 있다.

SVD(특이값 분해)는 임의의 행렬 A를 다음과 같이 세 개의 행렬로 분해합니다.
$$A = U\Sigma V^T$$

* $U$ 정의: $A$ 행렬이 $m \times n$ 크기라면, $U$는 $m \times m$ 크기의 직교행렬입니다.
* $\Sigma$ 정의: $\Sigma$는 $m \times n$ 크기의 대각행렬인데, NumPy에서는 대각원소만 1차원 배열 S로 반환합니다.
* $V^T$ 정의: $V$는 $n \times n$ 크기의 직교행렬이며, $V^T$는 이의 전치행렬입니다.
<!-- 여기서 $\Sigma$의 대각 원소들(특이값들)은 $A^TA$의 고유값들의 제곱근이다. -->

많은 응용에서는 전체 SVD 대신 압축된 형태인 "압축형 SVD"(compact SVD)나 "절단 SVD"(truncated SVD)를 사용한다.
$$A = U_r \Sigma_r V_r^T$$
여기서 $r$은 행렬의 랭크이고, $U_r$은 처음 $r$개의 왼쪽 특이벡터, $\Sigma_r$은 처음 $r$개의 특이값을 포함하는 $r \times r$ 대각행렬, $V_r$은 처음 $r$개의 오른쪽 특이벡터를 포함한다.

## 고유값 분해와의 비교
| 특이값 분해 (SVD) | 고유값 분해 (Eigendecomposition) |
|:-------------:|:------------------------:|
| 모든 행렬에 적용 가능 | 정방행렬에만 적용 가능 |
| $A = U\Sigma V^T$ | $A = PDP^{-1}$ |
| $U$, $V$는 직교행렬 | $P$는 일반적으로 직교행렬이 아님 |
| 특이값은 항상 실수이고 음이 아님 | 고유값은 복소수일 수 있음 |
| 수치적으로 안정적 | 일부 행렬에서는 불안정할 수 있음 |

```python
import numpy as np

# 행렬 정의
A = np.array([[4, 0, 3], [-3, -5, 0]])

# SVD 계산
U, S, Vt = np.linalg.svd(A)

print("U =\n", U)
print("S =\n", S)  # 특이값만 반환
print("V^T =\n", Vt)

# 원 행렬 재구성
S_matrix = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(S_matrix, S)
A_reconstructed = U @ S_matrix @ Vt

print("A_reconstructed =\n", A_reconstructed)
```
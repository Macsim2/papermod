---
title: "WFST 기반 디코딩의 세계: HCLG와 TLG의 내부 구조 탐색"
date: 2024-07-31T10:00:00+09:00
lastmod: 2024-07-31T10:00:00+09:00
draft: false
description: "음성 인식에서 사용되는 WFST 기반 디코딩 구조 HCLG와 TLG의 심층 분석"
tags: [
	"deeplearning",
	"ASR",
  "WFST", 
  "Kaldi",
  "WeNet",
  "speech recognition",
  "decoding"
]
categories: ["deeplearning"]
ShowToc: true
TocOpen: true
hidemeta: false
---

오래전부터 ASR 시스템을 개발한 사람이라면, kaldi라는 toolkit을 한번쯤 들어봤을 것이다. 대 NLP or Vision or multimodal 시대에서도 꿋꿋히 ASR을 정진해 나가는 dinal povey 교수님이시다..! 그가 2011년에 쓴 논문의 open source가 바로 kaldi toolkit 인데, 그 toolkit은 WFST(Weighted Finite State Transducer)이란 기법을 사용하고 뭐 겉으로 보기에도 그리 쉽게 이해되지 않는다. 

오늘은 **"이 복잡한 WFST 기반 디코딩 그래프는 도대체 어떻게 작동하는 걸까?"** 라는 궁금증을 가지고 이번 포스팅을 해보려고 한다. 특히 Kaldi의 HCLG 그래프나 WeNet의 TLG 디코딩 방식은 논문만 읽어서는 완전히 이해하기 어렵고, 코드를 직접 분석해야만 그 비밀이 드러나는 경우가 많은 것 같다.

이 글에서는 단순히 "WFST가 무엇인가"에 대한 기본 설명을 넘어, Kaldi와 WeNet에서 구현된 WFST 기반 디코딩 방식의 내부 구조와 작동 원리를 실제 예시와 함께 알아보고자 한다. <!--more-->

## WFST: 오토마타 이론에서 음성 인식까지

### 오토마타 이론과 WFST의 이론적 기반

WFST를 이해하기 위해서는 컴퓨터 과학의 근간이 되는 **오토마타 이론(Automata Theory)** 부터 시작해야 한다. 오토마타 이론은 추상적 기계와 계산 모델을 연구하는 이론으로, 1950년대 Noam Chomsky의 형식 언어 계층(Chomsky Hierarchy)과 함께 발전했다.

이 계층 구조에서 정규 언어(Regular Language)를 인식할 수 있는 가장 기본적인 모델이 **유한 상태 오토마타(Finite State Automata, FSA)** 이다. 유한 상태 오토마타는 다음과 같은 5개의 구성 요소를 가진다

$$A = (Q, \Sigma, \delta, q_0, F)$$

- $Q$: 유한한 상태 집합
- $\Sigma$: 입력 심볼의 유한한 집합(알파벳)
- $\delta$: 전이 함수 $\delta: Q \times \Sigma \rightarrow Q$
- $q_0 \in Q$: 초기 상태
- $F \subseteq Q$: 최종 상태들의 집합

유한 상태 오토마타는 입력 문자열을 읽고 그 문자열이 오토마타가 표현하는 언어에 속하는지 여부를 결정한다. 그러나 FSA는 입력만 처리할 뿐 출력을 생성하지는 않는다.

여기서 한 단계 더 발전한 것이 **유한 상태 변환기(Finite State Transducer, FST)** 이다. FST는 FSA와 유사하지만, 각 전이에서 입력 심볼을 읽고 출력 심볼을 생성한다.

$$T = (Q, \Sigma, \Delta, \delta, q_0, F)$$

- 추가된 $\Delta$는 출력 심볼의 집합
- 전이 함수는 $\delta: Q \times \Sigma \rightarrow Q \times \Delta$로 확장

FST는 언어 간의 관계를 모델링할 수 있다. 즉, 한 언어에서 다른 언어로의 변환을 표현할 수 있다.

마지막으로, **가중치 유한 상태 변환기(Weighted Finite State Transducer, WFST)** 는 FST에 가중치 개념을 추가한 것이다. 각 전이에는 입력, 출력 심볼 외에도 가중치가 할당되어 있다.

$$W = (Q, \Sigma, \Delta, \delta, \lambda, \rho, K)$$

- $\delta: Q \times \Sigma \times \Delta \times K \times Q$로 확장된 전이 관계
- $K$: 가중치가 속하는 반환(semiring) 구조
- $\lambda, \rho$: 초기/최종 상태의 가중치 함수

여기서 반환(semiring)은 두 연산(⊕, ⊗)이 정의된 대수 구조로, 음성 인식에서는 주로 대수적 연산을 모델링하는 tropical semiring을 사용한다. 이 구조에서는:
- ⊕ 연산은 min 또는 max (경로 선택)
- ⊗ 연산은 + (가중치 누적)

### WFST의 기본 개념

WFST(Weighted Finite State Transducer)는 한마디로 **입력 시퀀스를 출력 시퀀스로 변환하는 가중치 있는 유한 상태 기계**이다. 복잡하게 들리지만, 본질적으로는 노드(상태)와 간선(전이)으로 구성된 방향성 그래프(Directed Graph)이다.

기본 구성 요소를 다시 살펴보면:

- **상태(State)**: 그래프의 노드로, 특정 시점의 상태를 나타냄
- **전이(Transition)**: 하나의 상태에서 다른 상태로 이동하는 간선(Edge)
- **입력 심볼(Input Symbol)**: 전이가 발생하기 위해 필요한 입력
- **출력 심볼(Output Symbol)**: 전이가 발생할 때 생성되는 출력
- **가중치(Weight)**: 각 전이에 할당된 비용 또는 확률

형식적으로 WFST는 다음과 같이 표현할 수 있다.
$T = (Σ, Δ, Q, I, F, E, λ, ρ)$

여기서:
- $Σ$: 입력 심볼 집합
- $Δ$: 출력 심볼 집합
- $Q$: 상태 집합
- $I ⊆ Q$: 초기 상태 집합
- $F ⊆ Q$: 최종 상태 집합
- $E$: 전이 집합 $(q, i, o, w, q') ∈ Q × (Σ ∪ {\epsilon}) × (Δ ∪ {\epsilon}) × K × Q$
- $λ, ρ$: 초기/최종 가중치 함수

### 오토마타 관점에서의 WFST 연산

오토마타 이론의 관점에서 WFST의 핵심 연산들은 깊은 수학적 의미를 가진다:

1. **합성(Composition) ∘**: 함수 합성과 유사하게, 두 변환기를 연결하여 새로운 변환기를 생성한다. 이는 관계 대수학에서의 조인(join) 연산과 유사하다.

   $$A ∘ B = \{(x, z, w_1 \otimes w_2) | \exists y : (x, y, w_1) \in A \text{ and } (y, z, w_2) \in B\}$$

   이 연산은 범주론(Category Theory)에서 모르피즘의 합성으로도 해석될 수 있다.

2. **결정화(Determinization)**: 비결정성 오토마타(NFA)를 결정성 오토마타(DFA)로 변환하는 과정의 확장이다. 파워셋 구성(Powerset Construction) 알고리즘에 기반하며, 상태 폭발(state explosion)이 발생할 수 있지만 실행 효율성을 크게 향상시킨다.

3. **최소화(Minimization)**: Myhill-Nerode 정리에 기반한 최소 등가 오토마타 구성 알고리즘의 확장이다. 구별 불가능한 상태들을 병합하여 가장 작은 등가 오토마타를 만든다.

4. **가중치 푸시(Weight Pushing)**: 동적 프로그래밍 기법을 사용하여 가중치를 그래프의 앞쪽으로 재분배하는 연산으로, 이는 최단 경로 알고리즘과 밀접한 관련이 있다.

이러한 연산들은 단순한 그래프 조작을 넘어, 언어 이론과 정형 방법론(Formal Methods)의 깊은 개념들을 내포하고 있다.

### WFST의 음성 인식 적용 원리

이제 ASR에서의 WFST를 알아보자. WFST는 `음향 모델(Acoustic Model)의 출력`을 `단어 시퀀스`로 변환하는 복잡한 과정을 효율적으로 처리하는 데 사용된다. 이 과정에서 다음과 같은 변환 단계가 필요하다:

1. **음향적 특징 → 음소**: 음향 모델이 인식한 음소 확률
2. **음소 → 발음 변이**: 같은 음소 시퀀스가 다르게 발음될 수 있음
3. **발음 → 단어**: 발음 사전에 따른 단어 매핑
4. **단어 → 문장**: 언어 모델에 따른 문장 구성

이 모든 단계를 별도로 처리하면 계산 비용이 매우 높아진다. WFST는 이러한 변환 단계를 미리 **합성(Composition)** 하여 단일 그래프로 표현함으로써 효율적인 디코딩을 가능하게 한다.

```
[음향 모델 출력] → [WFST 기반 디코딩 그래프] → [단어 시퀀스]
```

오토마타 이론의 관점에서 보면, 음성 인식은 확률적 시퀀스(음향 특징)에서 또 다른 확률적 시퀀스(단어)로의 변환 문제이며, WFST는 이러한 확률적 변환을 수학적으로 엄밀하게 모델링하는 프레임워크를 제공한다.

WFST의 강력함은 다양한 지식 원천(음향 모델, 발음 사전, 언어 모델)을 하나의 통합된 프레임워크로 결합할 수 있다는 점에 있다. 이는 마치 다양한 언어(음향적 언어, 음소적 언어, 단어 언어)를 서로 변환하는 형식 문법을 구성하는 것과 같다.

### WFST와 확률적 문맥 자유 문법(PCFG)의 관계

음성 언어의 구조적 특성을 고려할 때, 왜 문맥 자유 문법(Context-Free Grammar, CFG)이 아닌 유한 상태 모델을 사용하는지 의문이 들 수 있다. Chomsky 계층에서 CFG는 FSA보다 더 표현력이 높기 때문이다.

실제로 초기 음성 인식 시스템 중 일부는 확률적 문맥 자유 문법(Probabilistic CFG)을 사용했다. 그러나 WFST는 다음과 같은 이유로 더 널리 채택되었다:

1. **계산 효율성**: WFST는 동적 프로그래밍 기법으로 효율적으로 최적화될 수 있다.
2. **통합 프레임워크**: 음향, 발음, 언어 모델을 단일 프레임워크로 통합할 수 있다.
3. **확장성**: 대규모 어휘 인식에서도 효율적으로 확장할 수 있다.
4. **풍부한 최적화 알고리즘**: 결정화, 최소화 등 다양한 최적화 기법을 적용할 수 있다.

또한, n-gram 언어 모델은 사실상 유한 상태 모델로, WFST와 자연스럽게 통합될 수 있다.

### WFST의 핵심 연산

WFST 프레임워크의 강력함은 다양한 연산을 통해 transducer를 조작하고 최적화할 수 있다는 점에 있다. 주요 연산으로는 아래와 같다.

- **합성(Composition) ∘**: 두 transducer를 연결하여 하나의 새로운 transducer 생성
  - $A ∘ B = \{(x, z, w_1 ⊗ w_2) | \exists y : (x, y, w_1) \in A \text{ and } (y, z, w_2) \in B\}$

- **결정화(Determinization)**: 각 상태에서 동일한 입력에 대해 정확히 하나의 전이만 갖도록 변환

- **최소화(Minimization)**: 동등한 기능을 유지하면서 상태와 전이의 수를 최소화

- **가중치 푸시(Weight Pushing)**: 가중치를 그래프의 앞쪽으로 재분배하여 탐색 효율성 향상

이러한 연산들을 활용하여 음성 인식 시스템의 디코딩 그래프를 효율적으로 구성하고 최적화할 수 있다.

![WFST 기본 연산 예시](/images/dl/wfst_operations.png)

## Kaldi의 HCLG: 상세 분석

이제 WFST라는 것의 정체를 조금 알게 되었으니 어떻게 Kaldi에 적용되었는지 알아보자. Kaldi는 현재까지도 많은 ASR 시스템의 기반이 되는 오픈 소스 툴킷으로, WFST 기반 디코딩의 표준을 확립했다고 볼 수 있다. Kaldi의 디코딩 그래프는 `HCLG.fst`라는 이름으로 불리는데, 이는 그래프를 구성하는 네 가지 주요 transducer의 합성을 의미한다.

```
HCLG = H ∘ C ∘ L ∘ G
```

각 컴포넌트를 하나씩 자세히 살펴보자.

### H transducer: HMM 상태 변환

`H` transducer는 **`HMM 상태(또는 PDF-ID)`를 `문맥 의존적 음소(CD-phone)`로 변환**하는 역할을 한다. 이는 음향 모델의 출력과 발음 모델 사이의 연결고리이다.

#### 입력과 출력 심볼

- **입력**: HMM 상태 ID (PDF-ID)
- **출력**: 문맥 의존적 음소 ID (triphone 또는 biphone)

#### 구체적인 예시

실제 Kaldi에서 `H.fst`의 일부를 살펴보면 아래와 예시 포맷을 발견할 수 있다.
'출발_상태(State) 도착_상태(State) 입력_심볼(Symbol) 출력_심볼(Symbol) [가중치]'
```
0 1 1 11001 0.4
1 2 2 11001 0.6
2 3 3 11002 0.3
```

이 예시에서:
- 상태 0에서 상태 1로의 전이는 입력 심볼 1 (HMM transition-ID 1)을 받아서(소비하고), 출력 심볼 11001 (문맥 의존적 음소 ID)을 내보내며, 이 전이의 가중치는 0.4입니다.
- 이런 식으로 음향 모델이 인식한 상태 시퀀스를 통해 음소 시퀀스로 변환

HMM 상태는 음향 모델 학습 시 결정된 결정 트리(decision tree)에 의해 클러스터링된 상태를 의미한다. 예를 들어 PDF-ID 1은 특정 문맥에서의 /a/ 음소의 시작 부분을 나타낼 수 있다.

```python
# H transducer 생성 의사 코드
for each context_dependent_phone in cd_phones:
    for each state in hmm_topology[context_dependent_phone]:
        add_transition(
            from_state=current_state,
            to_state=next_state,
            input_symbol=pdf_id[context_dependent_phone][state],
            output_symbol=context_dependent_phone,
            weight=transition_probability
        )
```

실제로 Kaldi에서는 HMM 토폴로지 파일을 통해 각 음소의 HMM 구조를 정의하며, 일반적으로 3-상태 left-to-right 구조를 사용한다.

### C transducer: 문맥 의존성 처리

`C` transducer는 **`문맥 의존적 음소(CD-phone)`를 `기본 음소(basephone)`로 변환**하는 역할을 한다. 이는 음향 모델에서 사용하는 풍부한 문맥 정보를 발음 사전에서 사용하는 기본 음소로 매핑한다.

#### 입력과 출력 심볼

- **입력**: 문맥 의존적 음소 ID (예: 좌우 문맥을 포함한 triphone)
- **출력**: 기본 음소 ID (context-independent phone)

#### 구체적인 예시 (`C.fst`)

```
0 1 11001 p_a 0.0
1 2 11002 a 0.0
2 3 11003 n 0.0
```

이 예시에서:
- FST의 상태 0에서 상태 1로 이동하는 전이는, 입력 심볼 11001 을 받으면 (즉, 음향 모델이 이 문맥 의존적 음소 ID에 해당하는 소리를 감지하면), 출력 심볼 p_a 를 내보냅니다.
이 변환의 가중치(비용) 는 0.0 입니다 (일반적으로 로그 확률 공간에서 0은 확률 1을 의미, 즉 비용 없음).
- FST의 상태 1에서 상태 2로 이동하는 전이는, 입력 심볼 11002 를 받으면 (다른 문맥을 가진 /a/ 소리일 수 있습니다), 출력 심볼 a 를 내보냅니다. 가중치는 0.0 입니다.
- 여기서 p_a는 위치 의존적 음소(position-dependent phone)로, 단어 시작 위치의 a를 의미

Kaldi에서는 위치 정보(시작, 중간, 끝)도 음소에 포함시켜 더 정확한 모델링을 가능하게 한다.

#### 왜 H.fst 로만 사용하지 않고 C.fst를 사용할까?
- 음향 모델은 미세한 소리 차이를 구분하기 위해 많은 문맥 의존적 음소를 사용하지만, 발음 사전에는 모든 문맥 변화를 기록하기 어려우므로, 이 둘 사이를 연결해주는 C.fst가 필요하다.

```python
# C transducer 생성 의사 코드
for each context_dependent_phone in cd_phones:
    base_phone = get_base_phone(context_dependent_phone)
    add_transition(
        from_state=current_state,
        to_state=next_state,
        input_symbol=context_dependent_phone,
        output_symbol=base_phone,
        weight=0.0  # 일반적으로 가중치 없음
    )
```

### L transducer: 발음 사전

`L` transducer는 **`음소 시퀀스`를 `단어`로 변환**하는 발음 사전(lexicon)을 나타낸다. 이는 단어의 다양한 발음 변이를 처리한다.

#### 입력과 출력 심볼

- **입력**: 기본 음소 ID 시퀀스
- **출력**: 단어 ID

#### 구체적인 예시

Kaldi에서 `L.fst`의 일부를 보면:

```
0 1 p_a <eps> 0.0
1 2 a <eps> 0.0
2 3 n <eps> 0.0
3 4 <eps> PANDA 0.0
```

이 예시에서:
- FST의 상태 0(시작 상태)에서 상태 1로 이동하는 전이는, 입력 심볼 p_a 를 받으면 (즉, 이전 단계 HC.fst에서 이 음소가 출력되면), 출력 심볼 <eps> (엡실론, epsilon)을 내보낸다.
- <eps>는 입실론(epsilon) 전이로, 입력 또는 출력 없이 상태 전이가 가능함을 의미

L transducer는 동일한 단어에 대한 여러 발음 변이를 포함할 수 있다:

```
# "TOMATO"의 여러 발음 변이
0 1 t_b <eps> 0.0
1 2 ax <eps> 0.0
2 3 m <eps> 0.0
3 4 ey <eps> 0.0
4 5 t <eps> 0.0
5 6 ow <eps> 0.0
6 7 <eps> TOMATO -0.69

0 1 t_b <eps> 0.0
1 2 ax <eps> 0.0
2 3 m <eps> 0.0
3 4 aa <eps> 0.0
4 5 t <eps> 0.0
5 6 ow <eps> 0.0
6 7 <eps> TOMATO -1.61
```
해석: FST는 상태 0에서 시작하여 차례대로 "t_b", "ax", "m", "ey", "t", "ow" 음소 시퀀스를 입력으로 받는다. <br>
이 과정 동안에는 계속 출력으로 <eps>(아무것도 없음)를 내보낸다. <br>
모든 음소가 입력된 후(상태 6 도달), 마지막 전이에서 더 이상 음소 입력 없이(<eps>) 최종 단어 "TOMATO"를 출력한다.

여기서 가중치(예: -0.69, -1.61)는 각 발음 변이의 확률을 나타낸다. 이 값은 로그값으로 값이 작을수록 더 높은 확률을 의미한다.

```python
# L transducer 생성 의사 코드
for each word in vocabulary:
    for each pronunciation in pronunciations[word]:
        current_state = start_state
        for phone in pronunciation:
            next_state = get_next_state()
            add_transition(
                from_state=current_state,
                to_state=next_state,
                input_symbol=phone,
                output_symbol="<eps>",
                weight=0.0
            )
            current_state = next_state
            
        # 마지막 음소 다음에 단어 출력
        final_state = get_next_state()
        add_transition(
            from_state=current_state,
            to_state=final_state,
            input_symbol="<eps>",
            output_symbol=word,
            weight=-log(pronunciation_probability)
        )
```

### G transducer: 언어 모델

`G` transducer는 **단어 시퀀스의 확률을 모델링**하는 언어 모델을 나타낸다. 일반적으로 n-gram 언어 모델을 WFST 형태로 변환한 것이다.

#### 입력과 출력 심볼

- **입력**: 단어 ID
- **출력**: 단어 ID (입출력이 동일)

#### 구체적인 예시

예를 들어 간단한 bigram 언어 모델의 일부:

```
0 1 <s> <s> 0.0
1 2 I I -1.1
1 3 YOU YOU -0.9
2 4 AM AM -0.7
2 5 HAVE HAVE -1.2
3 6 ARE ARE -0.5
3 7 WILL WILL -0.8
...
```

이 예시에서:
- 상태 1은 문장 시작(\<s\>) 다음에 올 수 있는 단어들을 나타냄
- "I" 다음에는 "AM"(-0.7)이나 "HAVE"(-1.2)가 올 수 있으며, "AM"이 더 높은 확률을 가짐
- 가중치는 로그 확률을 나타내므로, 값이 작을수록(0에 가까울수록) 더 높은 확률을 의미

실제 G transducer는 훨씬 더 복잡하며, 백오프(backoff) 아크를 포함하여 학습 데이터에서 보지 못한 n-gram도 처리할 수 있다.

```python
# G transducer 생성 의사 코드 (bigram 예시)
for each word1 in vocabulary:
    for each word2 in vocabulary:
        if bigram_exists(word1, word2):
            add_transition(
                from_state=state_for_history(word1),
                to_state=state_for_history(word2),
                input_symbol=word2,
                output_symbol=word2,
                weight=-log(probability(word2|word1))
            )
        else:
            # 백오프 처리
            backoff_weight = -log(backoff_weight(word1))
            add_transition(
                from_state=state_for_history(word1),
                to_state=backoff_state(word1),
                input_symbol="<eps>",
                output_symbol="<eps>",
                weight=backoff_weight
            )
```

### HCLG 합성 과정과 최적화

이제 네 개의 transducer가 어떻게 하나의 `HCLG.fst`로 합성되는지 살펴보자.

#### 합성 순서

합성 순서는 매우 중요하며, 일반적으로 다음과 같은 단계를 따른다.

1. **L ∘ G**: 먼저 발음 사전과 언어 모델을 합성하여 음소 시퀀스에서 단어 시퀀스로의 매핑을 구성
2. **C ∘ (L ∘ G)**: 문맥 의존적 음소와 위 결과를 합성하여 문맥 정보 통합
3. **H ∘ (C ∘ (L ∘ G))**: 마지막으로 HMM 상태 변환을 합성하여 최종 HCLG 그래프 생성

이 과정을 통해 음성 인식 시스템의 디코딩 그래프를 효율적으로 구성하고 최적화할 수 있다.

![HCLG 합성 과정 예시](/images/dl/hclg_composition.png)

#### 핵심 최적화 기법

단순히 네 개의 transducer를 합성하는 것으로는 효율적인 디코딩 그래프를 얻을 수 없다. Kaldi는 다양한 최적화 기법을 적용한다.

1. **결정화(Determinization)**: 각 단계 후 결정적 WFST로 변환하여 중복 경로 제거
2. **최소화(Minimization)**: 상태와 전이 수를 최소화하여 그래프 크기 축소
3. **가중치 푸시(Weight Pushing)**: 가중치를 그래프 앞쪽으로 재분배하여 탐색 효율성 향상
4. **에필론 제거(Epsilon Removal)**: 가능한 많은 입실론 전이를 제거하여 그래프 간소화

실제 Kaldi의 HCLG 그래프 생성 스크립트 일부를 보면,

```bash
# L과 G 합성
fstcomposecontext --context-size=$ncontxt --central-position=$central \
    --read-disambig-syms=$dir/phones/disambig.int \
    --write-disambig-syms=$dir/disambig_ilabels_${n_gram_order}.int \
    $dir/Ha.fst $dir/HLG.fst > $dir/HCLG.fst

# 최적화 적용
fstisstochastic $dir/HCLG.fst || echo "[info]: HCLG not stochastic."
fstdeterminizestar --use-log=true $dir/HCLG.fst | fstminimizeencoded | \
    fstpushspecial > $dir/HCLG_optimized.fst
```

#### 실제 HCLG 디코딩 과정

디코딩 과정은 음향 모델의 출력(HMM 상태 확률)과 HCLG 그래프를 결합하여 가장 가능성 높은 단어 시퀀스를 찾는 과정이다:

1. 음향 모델이 각 프레임에서 모든 HMM 상태의 확률(로그-가능도) 계산
2. 이 확률을 HCLG 그래프의 입력 심볼에 대한 비용으로 변환
3. 비터비 알고리즘을 사용하여 최소 비용 경로 탐색
4. 최종 경로에서 출력 심볼(단어)을 추출하여 인식 결과 생성

```cpp
// Kaldi 디코딩 의사 코드 (간략화됨)
for each frame in utterance:
    acoustic_costs = -log(acoustic_model_output)
    
    for each active_token in active_tokens:
        state = active_token.state
        for each arc in fst.arcs(state):
            input_symbol = arc.input_symbol
            if input_symbol != epsilon:
                cost = active_token.cost + arc.weight + acoustic_costs[input_symbol]
            else:
                cost = active_token.cost + arc.weight
                
            next_state = arc.next_state
            new_token = Token(next_state, cost, output_symbol=arc.output_symbol)
            add_to_next_tokens(new_token)
    
    // 토큰 정리 (pruning)
    prune_tokens(next_tokens, beam_width)
    active_tokens = next_tokens
```

#### 성능에 영향을 미치는 요소

HCLG 디코딩의 성능은 다양한 요소에 의해 영향을 받는다.

1. **그래프 크기**: 더 큰 그래프는 더 많은 메모리를 사용하고 검색 시간이 증가
2. **빔 너비(Beam Width)**: 넓은 빔은 정확도를 향상시키지만 속도를 저하
3. **언어 모델 크기**: 더 큰 n-gram 모델은 더 정확하지만 그래프 크기가 급증
4. **문맥 의존성 깊이**: triphone vs. quinphone 등 더 깊은 문맥은 더 정확하지만 복잡도 증가

실제 시스템에서는 이러한 요소들 간의 균형을 맞추는 것이 중요하다.

## WeNet의 TLG: E2E ASR을 위한 새로운 접근

End-to-End(E2E) ASR 시스템이 대세가 되면서, WFST 기반 디코딩도 이에 맞게 진화하고 있다. WeNet은 E2E ASR 시스템에서 WFST 디코딩을 효과적으로 적용한 대표적인 예이다.

### E2E ASR과 전통적 ASR의 디코딩 차이점

전통적인 HMM-GMM/DNN 기반 시스템과 E2E 시스템의 가장 큰 차이점은 `출력 단위`에 있다.

- **전통적 시스템**: 음향 모델은 HMM 상태(또는 음소) 확률 출력
- **E2E 시스템**: 모델은 `직접 토큰`(음소, 워드피스, 문자 등) 확률 출력

이러한 차이로 인해 디코딩 그래프 구성도 달라져야 한다. WeNet에서는 E2E 모델의 출력에 맞게 간소화된 `TLG` 디코딩 그래프를 사용한다.

### TLG 그래프의 구성 요소

`TLG` 디코딩 그래프는 다음 세 가지 컴포넌트의 합성으로 이루어진다:

```
TLG = T ∘ L ∘ G
```

각 컴포넌트를 자세히 살펴보자.

#### T transducer: 토큰 변환

`T` transducer는 **`E2E 모델의 출력 토큰`을 `발음 사전의 입력 단위`로 변환** 하는 역할을 한다. Kaldi의 H, C와 유사하지만 훨씬 단순하다.

##### 입력과 출력 심볼

- **입력**: E2E 모델의 출력 토큰 ID (예: 음소, BPE, 문자)
- **출력**: 발음 사전의 입력 단위 ID (보통 음소)

##### 구체적인 예시

만약 E2E 모델이 음소 단위로 출력한다면 T는 거의 항등 매핑에 가깝다:

```
0 1 AA AA 0.0
0 1 AE AE 0.0
0 1 AH AH 0.0
...
```

만약 BPE나 문자 단위라면 더 복잡한 매핑이 필요하다:

```
# BPE 토큰 "RE"를 음소로 변환하는 예
0 1 RE r 0.0
1 2 <eps> iy 0.0

# 문자 "X"를 음소로 변환하는 예
0 1 X eh 0.0
1 2 <eps> k 0.0
2 3 <eps> s 0.0
```

```python
# T transducer 생성 의사 코드
for each token in e2e_model_output_tokens:
    phoneme_sequence = token_to_phoneme_mapping[token]
    
    if len(phoneme_sequence) == 1:
        # 단일 음소인 경우 직접 매핑
        add_transition(
            from_state=0,
            to_state=1,
            input_symbol=token,
            output_symbol=phoneme_sequence[0],
            weight=0.0
        )
    else:
        # 여러 음소로 매핑되는 경우 (예: BPE 토큰)
        current_state = 0
        for i, phoneme in enumerate(phoneme_sequence):
            next_state = i + 1
            if i == 0:
                add_transition(
                    from_state=current_state,
                    to_state=next_state,
                    input_symbol=token,
                    output_symbol=phoneme,
                    weight=0.0
                )
            else:
                add_transition(
                    from_state=current_state,
                    to_state=next_state,
                    input_symbol="<eps>",
                    output_symbol=phoneme,
                    weight=0.0
                )
            current_state = next_state
```

#### L과 G transducer

WeNet의 `L`과 `G` transducer는 Kaldi의 것과 개념적으로 동일하다. 다만 E2E 시스템의 특성에 맞게 일부 조정이 있을 수 있다.

- **L transducer**: 음소 시퀀스를 단어로 변환
- **G transducer**: 단어 시퀀스의 확률 모델링

### TLG 디코딩 프로세스의 특징

WeNet의 TLG 디코딩은 다음과 같은 특징을 가진다:

#### 1. 토큰-기반 디코딩

E2E 모델은 프레임 단위가 아닌 토큰 단위로 출력을 생성한다. 이는 다음과 같은 중요한 차이를 가져온다:

- **더 적은 디코딩 스텝**: 프레임 수(보통 수백 개)보다 토큰 수(수십 개)가 훨씬 적음
- **더 높은 효율성**: 각 스텝에서 처리해야 할 데이터가 줄어듦

예를 들어, RNN-T 또는 CTC 기반 모델의 출력은 다음과 같을 수 있다:

```
# 토큰별 확률 분포 예시 (상위 3개만 표시)
Token 1: {"AH": 0.7, "IH": 0.2, "UH": 0.05, ...}
Token 2: {"W": 0.6, "HH": 0.3, "Y": 0.05, ...}
Token 3: {"AH": 0.5, "EH": 0.3, "AA": 0.1, ...}
...
```

#### 2. 디코딩 속도

```
| 모델 타입 | 실시간 비율(RTF) | 메모리 사용량 |
|---------|--------------|------------|
| HCLG    | 0.3-0.5      | 2-4GB      |
| TLG     | 0.1-0.2      | 300-500MB  |
```

TLG는 디코딩 스텝이 적고 그래프가 작아 더 빠른 속도와 적은 메모리 사용을 보인다.

##### 실제 구현 시 성능 고려사항

위 표는 일반적인 경향을 보여주지만, **실제 구현에서는 TLG 기반 디코딩이 항상 HCLG보다 빠르지 않을 수 있다.** 만약 직접 구현한 TLG 시스템이 예상보다 느리다면 다음과 같은 요인들을 고려해 볼 수 있다.

1.  **구현 및 최적화 수준:** Kaldi의 WFST 연산과 디코더는 고도로 최적화된 C++ 코드이다. 사용한 FST 라이브러리나 직접 구현한 디코더의 최적화 수준이 Kaldi보다 낮으면 속도 차이가 발생할 수 있다. 특히 그래프 결정화/최소화, 빔 서치 알고리즘(가지치기, 상태 관리 등)의 효율성이 중요하다.
2.  **E2E 모델 특성:** 문자 단위 출력 E2E 모델은 음소 기반보다 더 긴 토큰 시퀀스를 생성할 수 있어, FST 탐색 스텝 수가 늘어날 수 있다. 또한 E2E 모델 자체의 순방향 계산 속도도 전체 디코딩 시간에 영향을 준다.
3.  **디코딩 파라미터:** 비교 대상인 HCLG 시스템과 동일하거나 유사한 조건(빔 크기, 가지치기 임계값 등)에서 속도를 측정했는지 확인해야 한다. TLG에서 더 넓은 빔을 사용했다면 속도는 당연히 느려진다.
4.  **그래프 구성:** `T.fst`나 `L.fst`의 구성 방식이 비효율적이거나 그래프 크기가 과도하게 커졌을 수도 있다.

따라서 TLG의 이론적 장점에도 불구하고, 실제 성능은 구현 디테일, 사용된 모델, 비교 환경 설정에 따라 달라질 수 있다는 점을 유의해야 한다. 최적의 성능을 위해서는 디코딩 과정의 병목 현상을 분석하고, WeNet 등에서 사용하는 최적화 기법(예: 동적 그래프 확장, 히스토그램 프루닝)을 참고하는 것이 좋다.

#### 3. 인식 정확도

정확도 측면에서는 둘 다 비슷한 성능을 낼 수 있지만, 상황에 따라 차이가 있을 수 있다:

- **풍부한 학습 데이터 환경**: E2E 모델 + TLG가 일반적으로 우수
- **제한된 학습 데이터 환경**: HCLG 기반 시스템이 더 견고할 수 있음
- **도메인 특화 시나리오**: 둘 다 외부 언어 모델 통합으로 성능 개선 가능

#### 4. 실제 사용 시나리오

```
| 시나리오          | 권장 시스템   | 이유                                  |
|-----------------|------------|--------------------------------------|
| 모바일 기기       | TLG         | 작은 메모리 풋프린트, 빠른 응답 시간        |
| 서버 사이드 대규모 | 둘 다 가능    | 리소스 제약이 적음, 정확도 우선 선택       |
| 특수 도메인       | HCLG 유리   | 음향/언어 모델 분리로 도메인 적응 유연성 높음 |
| 저자원 언어       | HCLG 유리   | 언어 특화 지식을 명시적으로 통합 가능       |
```

## 마무리

WFST 기반 디코딩은 음성 인식 시스템의 핵심 구성 요소로 자리잡았으며, 전통적인 HCLG에서 E2E 시스템을 위한 TLG로 진화하고 있다. 둘 다 각자의 장단점이 있으며, 특정 응용 시나리오에 따라 적절한 선택이 필요하다.

미래의 디코딩 방향은 다음과 같을 것으로 예상된다.

1. **하이브리드 접근법**: E2E 모델의 강점과 WFST의 유연성을 결합한 방식
2. **동적 그래프 생성**: 실행 시간에 컨텍스트에 맞게 디코딩 그래프를 조정
3. **개인화된 디코딩**: 사용자 특화 언어 모델과 발음 변이를 포함
4. **멀티모달 통합**: 비디오, 텍스트 등 다양한 모달리티의 정보를 디코딩에 활용

어떤 방식을 선택하든, WFST 기반 디코딩의 핵심 원리를 이해하는 것은 ASR 시스템 개발자에게 필수적인 지식이다. 이 글이 복잡한 WFST 디코딩 그래프의 내부 구조를 이해하는 데 조금이라도 도움이 되었기를 바라며, 이 글을 마친다.

## References

* [1] Mohri, M., Pereira, F., & Riley, M. (2002). Weighted finite-state transducers in speech recognition. Computer Speech & Language, 16(1), 69-88.
* [2] Povey, D., et al. (2011). The Kaldi speech recognition toolkit. In IEEE Workshop on Automatic Speech Recognition and Understanding.
* [3] OpenFst Library: [http://www.openfst.org](http://www.openfst.org)
* [4] Kaldi HCLG Construction: [https://kaldi-asr.org/doc/graph_recipe_test.html](https://kaldi-asr.org/doc/graph_recipe_test.html)
* [5] WeNet: Production First and Production Ready End-to-End Speech Recognition Toolkit: [https://github.com/wenet-e2e/wenet](https://github.com/wenet-e2e/wenet)
* [6] Miao, H., Cheng, G., Gao, C., Zhang, P., & Yan, Y. (2020). Transformer-based online CTC/attention end-to-end speech recognition architecture. In ICASSP 2020.
* [7] Xu, H., et al. (2021). Improving Transformer-based Speech Recognition Using Unsupervised Pre-training. arXiv preprint arXiv:2101.02861.
* [8] Zhang, Z., et al. (2020). WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit. arXiv preprint arXiv:2102.01547. 
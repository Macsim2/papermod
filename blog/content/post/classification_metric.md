---
title: "Classification Metric"
date: 2023-02-21T21:10:51+09:00
draft: true
description: "Accuracy, Precision, Recall, F1-score"
tags: [
	"machine_learning",
	"classificaiton",
	"binary classification"
]
categories: ["machine_learning"]
math: true
ShowToc: true
TocOpen: true

---

ASR 분야에서 inference의 결과를 측정 할때 WER, CER 등의 metric을 사용한다. 이는 EditDistance 알고리즘을 이용한 것으로 정답 문자와 hypothesis 간에 유사도를 점수로 나타내는 방식이다.

인식(recognition), 분류(classification), 감지(detection)  task 에서 사용되는 방법인 accuracy, precision, recall 등에 대해서 알아보고자 한다.
비슷 하지만 다른  precision, recall, 그리고 FP, TN, TP, FN 등의 용어와 의미들 데이터의 해석까지 조금 정리 해보고자 한다.

## Binary Classification
-----------------
관련 내용을 찾아보니 워낙 폭넓게 쓰이는 용어, 개념이라 오늘은 우리가 주로 봐왔던 binary classification 문맥에서 얘기해보려 한다.
![bi_classification_map](/images/ml/biclassification_map.png)

$$ ACCURACY = \frac{TP+TN}{TP + TN + FP + FN} $$
$$ PRECISION = \frac{TP}{TP + FP} $$
$$ RECALL = \cfrac{TP}{TP + FN} $$

이것을 말로 조금 풀어내면,
$$ accuracy = \frac{정답}{모든\\,예측\\,case}\\quad모든\\,case\\,에\\,대하여\\,정답을\\,맞춘\\,비율$$<br>
$$ precision = \frac{정답인\\,positive\\,예측}{positive\\,예측} $$ precision을 정밀도라고도 부른다.<br>
$$ recall = \frac{정답인\\,positive\\,예측}{정답인\\,posivive\\,예측\\,+\\,오답인\\,negative\\,예측} $$ recall은 재현율이라고도 부른다.<br>

accuracy 야 일반적으로 많이 쓰기도 하고, 해석도 뭐 괜찮아 보인다. 모든 예측에 대해서 정답인 비율이니까 정확도를 말해주는게 맞는것 같다. 약간 시험에 대한 결과 같은 느낌이랄까<br>
그러면 precisio, recall 얘네들은 어떤 의미를 가지고 있는 것인지 궁금했다.<br>
precision, recall 등의 metric 은 어떻게 해석할 수 있을까? 해석을 이해하기 위해서 상황을 떠올리면 좀 더 이해가 쉽다.<br>

암환자의 진단과 실제 결과 상황을 아래의 map과 함께 이해 해보자.
![bi_classification_map_ex](/images/ml/biclassification_map_ex.png)

accuracy 는 정답 / 모든 case 이므로 8 + 3 / 8 + 2 + 1 + 3 => 11/14 ≈ 78.5%의 정확도를 가진다.<br>
그런데 문제는 **실제 암환자이지만 암진단을 받지 않은 경우도 있을 수 있다는 것이다.**<br>
암과 같이 다소 무거운 판정에서는 실제 환자를 진단해 내지 못하거나 실제 암환자가 아닌 사람에게 암진단을 하는 등의 무서운 일이 있을 수 있다는 것이다.<br>
특히 그런 상황에서는 accuracy의 효력은 줄어들게 된다.

## precision, recall
--------------------

### Precision 
precision은 암진단을 받은 사람들 중 실제 암 환자일 확률이다.<br>
TP / (TP + FP) = 8 / 8 + 1 => 8/9 ≈ 88.8% 의 정밀도를 나타낸다.<br>
암 진단의 자체의 신뢰성을 나타낼 수 있는 척도로 해석된다.

### Recall
recall 은 실제로 암환자인 사람을 진단할 확률이다.<br>
TP / (TP + FN) = 8 / 8 + 3 => 8/11 ≈ 72.7% 의 recall(재현율)을 나타낸다.<br>
실제 암 환자를 진단할 확률이라고 해석하면 되겠다.

### 해석
accuracy가 항상 좋지 않다는 것과 각 metric의 계산법이 다른 것도 알겠다. ~~그래서 어쩌라는 건가..?~~<br>
어디서 주워 듣기로 통계는 또 다른 해석학 이라는 얘기가 있다. 여기서도 비슷하다.<br>
결론적으로 이 질문의 답은 task에 달려 있다고 할 수 있다. **왜냐하면 FN(아니라고 예측했지만 실제로 맞는경우)때문이다.**<br>
두 가지 task가 있다고 가정해보자.<br>

하나는 광고 차단 task 가 있고, 다른 하나는 계속 해온 얘기인 암진단 task 가 있다.<br>
광고 차단 task에서는 살제로 광고가 아닌데 차단하는 경우(FP) 혹은 광고인데 차단하지 않는 경우(FN)가 광고 차단 프로그램 평판에 영향을 줄 수 있지만 피해가 크진 않다. 하지만 암진단 task인 경우 실제 암인데 판단하지 않는경우(FN)인 경우는 치명적이다. 사람의 목숨이 달려있을 수도 있는 문제이기 때문이다.<br>
이런 경우 FN을 중요하게 생각할 수 밖에 없고, recall 계산이 FN이 영향을 준다. 그래서 암진단 task의 경우는 recall 이 적합할 수 있는 것이다.<br>
**실제 데이터가 주어졌고 이 상황을 잘 분석해서 어떤 metric이 적합한지 디자인 하는게 중요하다 할 수 있겠다.**

## 보다 더 reasonable한 metric이 없을까?
precision, recall 어느것이 더 적합한 상황이 존재한다.<br>
상황에 덜 민감하면서 더 좋은 metric 방법은 없을까?

### F-measure



$$
\downarrow
$$

# references

* [1] https://en.wikipedia.org/wiki/Precision_and_recall 
* [2] 

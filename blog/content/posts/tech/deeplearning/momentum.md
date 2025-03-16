---
title: "Momentum in DeepLearning"
date: 2025-03-16T01:17:26+09:00
lastmod: 2025-03-16T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: 
- tech
tags: 
- DeepLearning
- optimizer
- SGD
description: "about Momentum"
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


## 모멘텀의 기본 개념과 직관적 이해

수학적 원리
경사 하강법의 기본 업데이트 규칙은 다음과 같다:
$$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$
여기서:
$\theta_t$는 현재 모델 파라미터
$\eta$는 학습률(learning rate)
$\nabla J(\theta_t)$는 현재 위치에서의 손실 함수의 기울기(gradient)
이 업데이트 규칙에서 스텝 크기를 결정하는 핵심 요소는 기울기의 크기이다. 손실 함수의 경사가 완만한 지역에서는 기울기 벡터 $\nabla J(\theta_t)$의 크기가 작아진다. 따라서:
경사가 가파른 지역: $|\nabla J(\theta_t)|$가 크다 → 큰 업데이트 스텝
경사가 완만한 지역: $|\nabla J(\theta_t)|$가 작다 → 작은 업데이트 스텝
직관적 이해: 등고선 지도 비유
산을 내려가는 등산객을 상상해보자:
가파른 경사면에서는 한 발짝 움직여도 많은 고도 변화가 생긴다
평평한 고원(plateau)에서는 한 발짝 움직여도 고도 변화가 거의 없다
딥러닝에서도 마찬가지로:
기울기가 큰 손실 함수 영역에서는 파라미터를 조금만 변경해도 손실 값이 크게 변한다
기울기가 작은 영역에서는 파라미터를 같은 크기로 변경해도 손실 값이 미미하게 변한다
구체적인 예시: 1차원 함수
간단한 1차원 함수 $f(x) = x^2$를 생각해보자:
$x = 10$일 때 기울기는 $f'(10) = 20$으로 매우 크다
학습률 $\eta = 0.1$일 때 업데이트: $x_{t+1} = 10 - 0.1 \times 20 = 8$
큰 스텝으로 2만큼 이동
$x = 0.1$일 때 기울기는 $f'(0.1) = 0.2$로 매우 작다
같은 학습률 $\eta = 0.1$일 때 업데이트: $x_{t+1} = 0.1 - 0.1 \times 0.2 = 0.08$
작은 스텝으로 0.02만큼만 이동
이 예시는 동일한 학습률에서도 기울기의 크기에 따라 스텝 크기가 크게 달라짐을 보여준다.

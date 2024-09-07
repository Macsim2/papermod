---
title: "LCS2"
date: 2024-08-28T01:17:26+09:00
lastmod: 2024-08-28T01:17:26+09:00
author: ["macsim"]
keywords: 
- 
categories: # 没有分类界面可以不填写
- 
tags: # 标签
- 
description: ""
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

# [BOJ 9252](https://www.acmicpc.net/problem/9252)

# 문제
LCS(Longest Common Subsequence, 최장 공통 부분 수열)문제는 두 수열이 주어졌을 때, 모두의 부분 수열이 되는 수열 중 가장 긴 것을 찾는 문제이다.
예를 들어, ACAYKP와 CAPCAK의 LCS는 ACAK가 된다.

# 입력
첫째 줄과 둘째 줄에 두 문자열이 주어진다. 문자열은 알파벳 대문자로만 이루어져 있으며, 최대 1000글자로 이루어져 있다.

# 출력
첫째 줄에 입력으로 주어진 두 문자열의 LCS의 길이를, 둘째 줄에 LCS를 출력한다.
LCS가 여러 가지인 경우에는 아무거나 출력하고, LCS의 길이가 0인 경우에는 둘째 줄을 출력하지 않는다.

--------------------------------------------

# 풀이
[LCS](https://www.acmicpc.net/problem/9251)과 다르게 LCS로 정해진 수열의 길이를 출력하는 것이 아닌 아무거나 하나를 구하는 조건이 더해졌다.

문제를 보고 어떻게 할지 고민을 조금 했는데 더 간단한 풀이방법이 있어서 기록한다.
LCS 문제는 2차원 배열을 만들고, 원소의 data type은 int형을 사용하여 LCS 길이를 구할 수 있었다.
그런데 LCS2는 2차원 배열을 사용하되, 원소의 data type를 string 을 사용해 LCS를 직접 구하는 방법을 사용할 수 있다.


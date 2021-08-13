---
title: "[boostcamp AI Tech] 학습기록 day02"
date: 2021-08-03 23:20:18 -0400
categories:
---

# AI Math
## 1. 벡터
* 벡터란?  
    * 수학적으로 크기와 방향을 가지는 물리량으로 숫자를 원소로 가지는 리스트를 의미합니다.
    * 공간에서의 한점으로 원점에서의 상대적 위치를 표현합니다.
    $$
    {\mathbf{x}}=\left[\begin{array}{c}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{d}
    \end{array}\right] \quad {\mathbf{x}^{\top}}=\left[x_{1}, x_{2}, \ldots, x_{d}\right]
    $$
* 벡터의 연산  
    * 벡터의 덧셈과 뺄셈: 벡터의 덧셈은 한 점에서 다른 한점의 상대적 위치 이동을 의미합니다.

    * 스칼라곱: 벡터에 스칼라값을 곱해주면 벡터의 방향은 변하지 않고 길이만 변하게 됩니다. 
    $$
    \alpha \mathbf{X}
    $$

    * 성분곱: 같은 모양의 벡터는 같은 성분끼리 곱하는 성분곱을 할 수 있습니다.
    $$
    \mathbf{x}=\left[\begin{array}{c}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{d}
    \end{array}\right] \quad \mathbf{y}=\left[\begin{array}{c}
    y_{1} \\
    y_{2} \\
    \vdots \\
    y_{d}
    \end{array}\right] \quad \mathbf{x} \odot \mathbf{y}=\left[\begin{array}{c}
    x_{1} y_{1} \\
    x_{2} y_{2} \\
    \vdots \\
    x_{d} y_{d}
    \end{array}\right]
    $$



* 벡터의 노름
    * 벡터의 노름(norm)은 원점에서 벡터의 거리를 측정하기 위한 수단으로 사용됩니다. 많이 사용하는건 $L_{1}$이나 $L_{2}$ 노름을 많이 사용하며 아래와 같이 정의됩니다.
    $$L_{1}=\|\mathbf{x}\|_{1}=\sum_{i=1}^{d}\left|x_{i}\right|$$
    $$L_{1}=\|\mathbf{x}\|_{2}=\sqrt{\sum_{i=1}^{d}\left|x_{i}\right|_{s}^{2}}$$
    * 노름에 따라 기하학적 성질이 달라지기 때문에 머신러닝 등에 사용할 때 두 노름의 성질을 고려하여 사용하여야 합니다.

* 벡터의 내적
    * 벡터의 내적은 물리학적으로 정사영된 벡터의 길이와 연관이 있습니다.
    * 정사영된 벡터의 길이를 다른 벡터의 길이와 곱하는 것으로 물리학적으로 일(W)와 연관이 있습니다.
    $$\langle\mathbf{x}, \mathbf{y}\rangle=\|\mathbf{x}\|_{2}\|\mathbf{y}\|_{2} \cos \theta$$



## 2. 행렬
* 행렬이란?
    * 행렬(matrix)는 숫자를 원소로 가지는 2차원 배열입니다.

    * 행렬은 연립일차 방정식을 풀이하기 위해서 행렬식을 고안하면서 나오게 되었습니다.
    $$
    \mathbf{X}=\left[\begin{array}{c}
    \mathbf{x}_{1} \\
    \mathbf{x}_{2} \\
    \vdots \\
    \mathbf{x}_{n}
    \end{array}\right]=\left[\begin{array}{cccc}
    x_{11} & x_{12} & \cdots & x_{1 m} \\
    x_{21} & x_{22} & \cdots & y_{2 m} \\
    \vdots & \vdots & & \vdots \\
    x_{n 1} & x_{n 2} & \cdots & x_{n m}
    \end{array}\right] 
    $$

    * 행(row)와 열(column)으로 인덱스를 가지고 각각의 행열의 수에 따라 $n \times m$ 행렬이라고 부릅니다.

    * 벡터가 한공간의 점을 의미한다면 행렬은 이러한 벡터가 여러개 모여있으므로 한 공간에서 여러개의 점들을 한번에 나타낼 수 있습니다.


* 행렬의 기본 연산
    * 기본적으로 행렬의 덧셈과 뺄셈, 스칼라곱, 성분곱은 2차원으로 확장을 했을 뿐이지 벡터와 똑같습니다.

    * 행렬의 덧셈과 뺄셈: 같은 모양을 가지는 행렬은 성분끼리 덧셈이나 뺄셈을 할 수 있습니다.
    $$
    \mathbf{X} \pm \mathbf{Y}=\left[\begin{array}{cccc}
    x_{11} \pm y_{11} & x_{12} \pm y_{12} & \cdots & x_{1 m} \pm y_{1 m} \\
    x_{21} \pm y_{21} & x_{22} \pm y_{22} & \cdots & x_{2 m} \pm y_{2 m} \\
    \vdots & \vdots & & \vdots \\
    x_{n 1} \pm y_{n 1} & x_{n 2} \pm y_{n 2} & \cdots & x_{n m} \pm y_{n m}
    \end{array}\right]
    $$
    * 스칼라곱:각 원소에 스칼라값을 곱해줍니다.

    * 성분곱: 각 성분끼리 곱해주는 연산을 합니다.
    $$
    \mathbf{X} \odot \mathbf{Y}
    =\left[\begin{array}{cccc}
    x_{11} y_{11} & x_{12} y_{12} & \cdots & x_{1 m} y_{1 m} \\
    x_{21} y_{21} & x_{22} y_{22} & \cdots & x_{2 m} y_{2 m} \\
    \vdots & \vdots & & \vdots \\
    x_{n 1} y_{n 1} & x_{n 2} y_{n 2} & \cdots & x_{n m} y_{n m}
    \end{array}\right]
    $$

* 행렬곱
    * 행렬곱(matrix multiplication)은 위의 기본연산과 달리 두 행렬에서 각각 행벡터와 열벡터를 가지고 내적을 하여 계산합니다.

    * 행렬곱을 한 $x_{ij}$의 성분은 첫번째 행렬에서 i번째 행벡터와  2번째 행렬에서 j번째 열벡터를 내적한 값을 가집니다.

    * $m \times n$ 행렬 $\mathbf{X}$와 $n \times p$ 행렬 $\mathbf{Y}$의 행렬곱은 다음과 같다.
    $$
    \mathbf{X}\mathbf{Y}=\mathbf{Z}=\left[\begin{array}{cccc}
    \sum_{k} x_{1 k} y_{k 1} & \sum_{k} x_{1 k} y_{k 2} & \cdots & \sum_{k} x_{1 k} y_{k p} \\
    \sum_{k} x_{2 k} y_{k 1} & \sum_{k} x_{2 k} y_{k 2} & \cdots & \sum_{k} x_{2 k} y_{k p} \\
    \vdots & \vdots & & \vdots \\
    \sum_{k} x_{m k} y_{k 1} & \sum_{k} x_{m k} y_{k 2} & \cdots & \sum_{k} x_{m k} y_{k p}
    \end{array}\right]
    $$


* 전치행렬
    * 행렬의 행과 열의 인덱스를 서로 바꾼 행렬을 의미합니다.
    $$
    \mathbf{X}^{\top}=
    \left[\begin{array}{cccc}
    x_{11} & x_{21} & \cdots & x_{n 1} \\
    x_{12} & x_{22} & \cdots & x_{n 2} \\
    \vdots & \vdots & & \vdots \\
    x_{1 m} & x_{2 m} & \cdots & x_{n m}
    \end{array}\right]
    $$

* 역행렬
    * 역행렬 (inverse matrix)은 원래 행렬과 곱했을 때 항등행렬(identity matrix) $\mathbf{I}$ 가 나오는 행렬을 의미한다.
    $$\mathbf{A} \mathbf{A}^{-1}=\mathbf{A}^{-1} \mathbf{A}=\mathbf{I}$$

    * 역행렬에는 조건이 있는데 행과 열의 길이가 같아야 하고 행렬식 (determinant)이 존재해야 한다.

* 유사역행렬
    *유사역행렬 (pseudo-inverse) 또는 무어-펜로즈 (Moore-Penrose) 역행렬이라고 부르며 일반적으로 역행렬의 조건이 맞지 않을 댸 역행렬 대신 사용한다.
    $$
    \begin{aligned}
    &n \geq m \text {인 경우} 
    \mathbf{A}^{+}=\left(\mathbf{A}^{\top} \mathbf{A}\right)^{-1} \mathbf{A}^{\top}, 
    &n \mathbf{A}^{+} \mathbf{A}=\mathbf{I}
    \\
    &n \leq m \text {인 경우}
    \mathbf{A}^{+}=\mathbf{A}^{\top}\left(\mathbf{A} \mathbf{A}^{\top}\right)^{-1},
    &n \mathbf{A A}^{+}=\mathbf{I}
    \end{aligned}
    $$

    $$
    \begin{aligned}
    
    
    \end{aligned}
    $$

* 행렬의 이해
    * 행렬은 벡터 공간에서 사용되는 연사자로 어떤 벡터를 다른 차원으로 선형변환(linear transform)하는 것으로 이해할 수 있습니다.
    $$
    \left[\begin{array}{c}
    z_{1} \\
    z_{2} \\
    \vdots \\
    z_{n}
    \end{array}\right]=\left[\begin{array}{cccc}
    a_{11} & a_{12} & \cdots & a_{1 m} \\
    a_{21} & a_{22} & \cdots & a_{2 m} \\
    \vdots & \vdots & & \vdots \\
    a_{n 1} & a_{n 2} & \cdots & a_{n m}
    \end{array}\right]\left[\begin{array}{c}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{m}
    \end{array}\right]
    $$
    * 이를 이용하여 연립방정식을 풀거나 선형모델로 해석하여 선형회귀식의 문제를 풀 수 있습니다.

# 과제
과제가 여러개 나왔는데 오늘 필수과제 1,2,3을 수행하였습니다.
## 과제 1
첫번 째 과제는 기본적인 계산기를 만드는 숙제였는데 numpy와 파이썬에서 기본으로 제공하는 list를 이용하여 둘다 구현해보았습니다. numpy로는 대부분 구현되어있는 함수여서 list로도 간단하게 짜서 구현해보았습니다.
## 과제 2, 3
2, 3의 과제 python  string을 이용해 보는 것이었는데 join(), split()등 파이썬 내장함수를 최대한 이용하여 최대한 짧게 해보려 노력하였다.

# [피어세션](https://hackmd.io/IXc2P0IwQXaqnThcWIq4lg)

# 후기
이번 주는 선행학습에 대한 복습때문인지 내용이 너무 많아서 모든 내용을 정리하기는 벅차다고 생각했다. 그래서 앞으로 AI에 필요한 수학 개념들을 나누어 복습하고 정리하려고 계획하였다. 나누어 정리하고 파이썬은 강의로 복습하여 이번주 달 끝내보는 것으로...
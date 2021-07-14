# Spoiler-Detection

## 딥러닝으로 영화 리뷰 속 스포일러 찾기

프로젝트 기간: 2021/01/06 ~ 2021/01/12

회고 글: https://velog.io/@gayeon/딥러닝으로-영화-리뷰-속-스포일러-찾기

<br/>

## 개요

  스포일러가 포함된 영화 리뷰를 찾아내는 딥러닝 모델을 만드는 프로젝트입니다. 리뷰 기능이 있는 OTT 서비스나 영화 큐레이션 서비스에서 사용자가 지속적으로 스포일러가 포함된 리뷰를 보게 된다면 스포일러를 피하고자 해당 서비스에 접속하는 것 자체를 꺼리게 될 수도 있습니다. 따라서, 고객들이 스포일러가 포함된 리뷰 때문에 다른 서비스로 이동하는 것을 방지하기 위해서 자연어 처리를 사용하여 스포일러가 포함된 리뷰를 처리할 수 있는 모델을 만들고자 하였습니다.
  
  사용할 수 있는 시간과 컴퓨터 자원에 한계가 있었지만 다행히 해결하고자 하는 문제가 이진 분류로 간단하였습니다. 따라서 무조건 자연어 처리 분야의 최신 모델을 적용하기보다는 두 문서의 유사도를 판단하는 모델 중에서 비교적 가벼운 모델을 조사하였으며, 그 과정에서 ManhattanLSTM(이하 MaLSTM)을 알게 되었습니다. 해당 모델을 응용하여 임베딩에는 [Word2Vec을 사용하여 사전 훈련된 벡터](https://code.google.com/archive/p/word2vec/)를, 모델 구조에는 Siamese neural network 구조와 Convolution, LSTM, Fully connected layer를 사용해서 모델을 생성하였습니다.

<br/>

 ## 사용한 데이터셋
 
IMDB Spoiler Dataset: Can you identify which reviews have spoilers to improve user experience?
- Rishabh Misra, 2019. 05. <br/>
- doi: 10.13140/RG.2.2.11584.15362 <br/>
- https://rishabhmisra.github.io/publications
- https://www.kaggle.com/rmisra/imdb-spoiler-dataset?select=IMDB_reviews.json (License - CC0: Public Domain)

위 데이터셋은 영화 제목, 상영 시간, 장르, 줄거리 등 영화 정보에 대한 데이터와 리뷰별 작성자, 해당하는 영화의 코드, 리뷰 내용 등 리뷰 정보에 대한 데이터를 가지고 있습니다. 이 중에서 영화의 플롯 요약본과 리뷰를 사용해서 프로젝트를 진행하였습니다.

<br/>

## 프로젝트 수행 과정

- 기준 모델
  - 모든 리뷰에 대해서 '스포일러를 포함하고 있지 않다.라고 예측하는 것
  - Validation dataset에 대한 accuracy: 약 0.7693
  - Validation dataset에 대한 f1 score: 0

<br/>

- 모델 생성

![image](https://user-images.githubusercontent.com/70365836/122967695-b3977e00-d3c5-11eb-80f8-b43dac3dcaf4.png)

<br/>

- Step 2 결과 생성된 모델의 validation dataset에 대한 accuracy, f1 score (소수점 다섯째 자리에서 반올림)

| Siamese neural network 부분에 사용된 layer | Accuracy | F1 score |
|:----------------------------------------|:---------|:---------|
| (MaLSTM)                                | 0.7731   | 0.2635   |
| LSTM                                    | 0.7844   | 0.2801   |
| GRU                                     | 0.7889   | 0.3153   |
| BiLSTM                                  | 0.7867   | 0.2972   |
| BiGRU                                   | 0.7910   | 0.3438 |
| Conv1D + LSTM | 0.7932 | 0.4079 |
| Conv1D + GRU | 0.7918 | 0.3719 |
| Conv1D + BiLSTM | 0.7923 | 0.4123 |
| Conv1D + LSTM + LSTM | 0.7969 | 0.3530 |
| Conv1D + BiLSTM + LSTM | 0.7955 | 0.3689 |
| Conv1D + LSTM + BiLSTM | 0.7967 | 0.3926 |


<br/>

## 최종 모델

- 모델 구조

![image](https://user-images.githubusercontent.com/70365836/125570194-b68bd453-03bd-4ff6-bc41-66bf35e5cf84.png)

- Test dataset에 대한 결과
  - Accuracy: 약 0.8025
  - F1 score: 약 0.4000

<br/>

## 배운 점

> "주어진 상황에서 선택할 수 있는 최선의 방법을 찾아야 한다."

  프로젝트 기간이 일주일로 매우 짧았고, 사용할 수 있는 컴퓨터 자원에는 한계가 있었습니다. 따라서, 최신 기술만을 고집하기보다는 현재의 상황과 해결해야 할 문제를 분석하고 그에 맞는 방법을 찾고자 노력하였고, 그 덕분에 기간 내에 결과물을 도출할 수 있었습니다. 이를 통해서 최신 기술이 항상 최선의 방법인 것은 아니라는 것, 그리고 주어진 상황을 빠르게 파악하고 그에 맞는 방법을 찾는 게 중요하다는 것을 배울 수 있었습니다.

<br>

### 참고 문헌

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1746–1751.
    https://arxiv.org/abs/1408.5882  
- Mueller, J., & Thyagarajan, A. (2015). Siamese Recurrent Architectures for Learning Sentence Similarity. AAAI-16. Arizona, USA.
- Pontes, E., & Huet, S., & Linhares, A., Torres-Moreno, J. (2018). Predicting the Semantic Textual Similarity with Siamese CNN and LSTM. *Traitement Automatique des Langues Naturelles (TALN)*, 1, 311–320.
    https://arxiv.org/abs/1810.10641

<br/>

### 참고 게시글

- [One Shot Learning with Siamese Networks using Keras](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d)
- [How to predict Quora Question Pairs using Siamese Manhattan LSTM](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)
- [클린봇 2.0: 문맥을 이해하는 악성 댓글(단문) 탐지 AI](https://d2.naver.com/helloworld/7753273)

<br/>

이 페이지에 첨부된 이미지에는 네이버에서 제공한 나눔글꼴이 사용되었습니다.

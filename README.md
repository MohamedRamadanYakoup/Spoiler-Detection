# Spoiler-Detection

## 딥러닝으로 영화 리뷰 속 스포일러 찾기

프로젝트 기간: 2021/01/06 ~ 2021/01/12

회고 글: https://velog.io/@gayeon/딥러닝으로-영화-리뷰-속-스포일러-찾기

<br/>

## 개요

자연어 처리를 이용해서 스포일러 영화 리뷰를 찾아내는 프로젝트이다. 해결하고자 하는 문제가 간단하다는 점, 사용할 수 있는 컴퓨터 자원에 한계가 있다는 점을 고려하여 두 문서의 유사도를 판단하는 모델 중에서 비교적 가벼운 모델인 Manhattan LSTM(이하 MaLSTM)을 사용하였다. Word2Vec을 이용해서 영화의 플롯 요약본과 리뷰를 임베딩 한 후, MaLSTM과 MaLSTM의 구조에서 착안하여 FC, CNN, LSTM을 사용한 Siamese Network 모델을 만들었다.

<br/>

 ## 사용한 데이터셋
 
IMDB Spoiler Dataset: Can you identify which reviews have spoilers to improve user experience?
- Rishabh Misra, 2019. 05. <br/>
- doi: 10.13140/RG.2.2.11584.15362 <br/>
- https://rishabhmisra.github.io/publications
- https://www.kaggle.com/rmisra/imdb-spoiler-dataset?select=IMDB_reviews.json

<br/>

## 최종 결과

- 모델 구조

<img src="https://user-images.githubusercontent.com/70365836/119447678-3ac3e880-bd6b-11eb-8115-c1d754bef0c8.png" width="400" height="700">

- accuracy: 약 0.8025
- f1 score: 약 0.4000

<br/>

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

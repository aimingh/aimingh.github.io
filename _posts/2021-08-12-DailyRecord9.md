---
title: "[boostcamp AI Tech] 학습기록 day09 (week2)"
date: 2021-08-12 23:52:43 -0400
categories:
---

# Deep Learning Basic
- 작업 중
## Sequrntial Model 
* Sequrntial Model
    * Naive sequence model
    * Autoregressive model
    * Markov model (first-order )
    * Latent autoregressive model

* RNN (Recurrent Neural Network)

* LSTM (Long Short Term Memory)

* GRU (Gated Recurrent Unit)


# 과제
## 필수과제4 Multi-Headed Attention
* Attention에 대한 개념을 실제로 어떻게 구현하는지 알 수 있는 과제였습니다.
```
class ScaledDotProductAttention(nn.Module):
    def forward(self,Q,K,V,mask=None):
        d_K = K.size()[-1] # key dimension
        scores = Q.matmul(K.transpose(-2,-1)) / np.sqrt(d_K) # FILL IN HERE
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attention = F.softmax(scores,dim=-1)
        out = attention.matmul(V)
        return out,attention
```
* attention의 score를 코드로 구현하는 부분이었다. 백터 내적을 이용하여 곱해주고  크기의 루트를 씌워 나누어주었습니다.
```
def forward(self,Q,K,V,mask=None):
    ...
    # Multi-head split of Q, K, and V (d_feat = n_head*d_head)
    Q_split = Q_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
    K_split = K_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
    V_split = V_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
    # Q_split: [n_batch, n_head, n_Q, d_head]
    # K_split: [n_batch, n_head, n_K, d_head]
    # V_split: [n_batch, n_head, n_V, d_head]

    # Multi-Headed Attention
    d_K = K.size()[-1] # key dimension
    scores = torch.matmul(Q_split,K_split.permute(0,1,3,2)) / np.sqrt(d_K) # FILL IN HERE
    ...
```
* Multi-Headed Attention (MHA)인데 위와 같은 수식이다. 다만 multi head이기 때문에 transpose를 해주어야 하는데 permute()을 이용하여 차원을 맞교환 하였다.


# [피어세션](https://hackmd.io/@ai17/SkQIAHMeY)

# 후기
특강이나 오피스아워와 개인사정으로 쉽게 정리할 수 없는것이 너무 아쉬웠다. 바로바로 정리하는것이 학습정리가 잘되는데 시간이 부족했던 하루였다.
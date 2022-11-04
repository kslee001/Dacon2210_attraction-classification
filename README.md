# DACON_2210 : 관광데이터 AI 경진대회

## 후기

- 2022.11.05  
- 많은 걸 배운 대회였다. 이 대회에서 multimodality라는 걸 처음 접했는데 앞으로도 이 분야를 계속 공부해보고 싶다는 생각이 들만큼 매력적인 분야였다. 대회에 쓰인 데이터도 image와 text, 딥러닝 단골 소재여서 참고할 리터러쳐도 풍부했다. 시간의 한계 때문에 생각해 본 모든 방법을 실험해 보지 못한 점이 아쉬울 뿐이다.
- 대회의 task가 처음에 생각한 것보다 더 흥미로웠다. 그래서 예상했던 것보다 너무 많은 시간을 쏟게 됐다. 취미인만큼 학교 수업에는 지장이 안 가도록 하고 싶었는데 결국 한 과목에서 약간의 지장이 생겨버리고 말았다. ㅎ.. 그러나 이 부분에 대한 후회는 없다
- 어떻게든 성능 올려보겠다고 온갖 논문 다 읽어본 것 같다. 학교 오가는 길에 버스에서 논문 읽는게 그렇게 즐거운 일인 줄 몰랐다. 지금 생각해보면 약간 미쳐있었던 듯..
- 최종 성적이 좀 아쉽긴 하다. 다음에는 더 잘해서 꼭 상금 타야지!

---

## 1. 대회 개요

- 주최 : 한국관광공사, 씨엘컴퍼니
- 주관 : 데이콘
- 대회기간 : 2022.09.19 ~ 2022.10.31
- Task
    - i**mage + text multimodal classification**
    - 이미지와 텍스트를 활용하여 해당 관광명소의 소분류(cat3)를 분류
- Data
    - train data
        - 16989개
        - image : 해당 장소를 촬영한 사진
            - 화질과 크기가 다양하여 preprocessing 필요
            - 결측값 없음
        - text : 해당 장소에 대한 한글 설명 (”overview”)
            - 결측값 존재
        - cat1, cat2 : 해당 장소의 대분류, 중분류 정보
    - test data
        - 7280개
        - image, text만 주어지며 cat1, cat2는 주어지지 않음
- 상금 : 총 1000만원
    - 1위 500만원
    - 2위 300만원
    - 3위, 4위 100만원

## 2. 대회 결과

- 최종 성적
    - Public  :
        - **Weighted F1 score : 0.84684  ( 53 / 290 )**
            - 1위 : 0.87152
    - Private :
        - **Weighted F1 score : 0.84365  ( 61 / 290 , top 21% )**
            - 1위 : 0.8658

## 3. 진행 과정

### 전처리

- highly imbalanced data
    - 음식점 (특히 한식), 야영지 category가 심각하게 많은 반면 (3000개, 1500개 등)
    - 단 2개밖에 없는 category도 상당 수 존재
    - → data augmentation의 필요성
- Image
    - train용 image data의 width, hight 크기는 대부분 250 이하
    - image size
        - 처음에는 inception net, (pretrained) efficient net의 paper에서 제시된 이미지 크기에 맞추기 위해 298 x 298 로 resize하였음
            - 억지로 크기를 늘려야 함 → 성능 저하
        - 이후 **284 x 284 로 변경 후 256 x 256 center crop** 적용
- Text
    - 처음에는 baseline code에서 제시한 대로, sklearn의 count vectorizer를 그대로 이용
    - 이후 konlpy 라이브러리의 Okt 토크나이저를 이용하여 [명사, 형용사, 동사] 만 추출하여 vectorize하였음
    - 최종적으로는 klue-roberta-large 사전학습 모델의 토크나이저와 모델을 이용하였음
        - 문장의 길이(토큰 개수)는 256으로 제한하고 max length padding 과 truncation 적용함

### 데이터 증강

- **Image**
    - albumentation이 제공하는 augmentation function들을 이용함
        - (반드시 적용)
            - Resize (W, H)
                - 짧은 쪽의 길이를 284에 맞추고, 긴 쪽의 길이를 비례적으로 변경함
                - 예를 들어 원본이 (512 x 384)인 이미지인 경우,  Resize 과정에서 (378 x 284)로 변경
                - 정보 손실 최소화를 위함
            - RandomCrop (284. 284)
        - (이 중 하나)
            - HorizontalFlip
            - Rotate
        - (이 중 하나)
            - MotionBlur
            - MedianBlur
            - Blur
        - (이 중 하나)
            - ISONoise
            - GaussNoise
            - RandomBrightnessContrast
            - ColorJitter
    - 기존의 이미지를 transform하는 것이 아니라 “완전히 다른” 이미지를 만들어 내는 것이 목적이므로, hard transformation을 진행
    - augmented되어 새로 생성된 이미지들은 pickle을 이용하여 저장
        - → 앞으로는 h5py를 이용할 것
- **Text**
    - 초기에는 별다른 augmentation을 수행하지 않음
    - 유의어 사전을 이용하여 augment 하는 방법을 시도하였으나 효과적이지 못하였음
    - 최종적으로는 KorEDA 라이브러리를 조금 수정하여 이용함
        - [https://github.com/catSirup/KorEDA](https://github.com/catSirup/KorEDA)
- **Replica problem with a single random seed**
    - 50개, 100개, 250개 등의 count 기준을 정해두고, 이 개수에 못미치는 카테고리에 대해서만 augmentation을 수행하였음
    - 그런데 데이터가 2개밖에 없는 경우, 같은 random seed로 augmentation을 진행하면 수십개의 “복제본”이 발생하는 문제가 발생함
        - 같은 random seed에 따라 똑같은 샘플이 뽑히고, 똑같은 augmentation이 반복되어 수행되기 때문
    - 이를 해결하기 위해 다음과 같은 코드를 삽입
    
    ```python
    original_seed = CFG["SEED"]
    
    new_seed = 0
    for i in range(augmentation_count):
    	new_seed +=1
    	cur_seed = original_seed + new_seed  
    
    	*random sampling code [ cur_seed ] ...*
    	*augmentation code    [ cur_seed ] ...*
    	... 
    ```
    
    - 이는 “재현 가능”하면서도 “매 iteration 마다 다른” random seed를 구현하기 위함

### 모델 선정

- Image
    - 청경채 대회와 마찬가지로, Deep learning 모델을 직접 구현하며 공부하는 것이 1차 목표
        - Inception network (google net) 구현하여 train에 이용
    - 그러나 성능이 좋지 않아 널리 쓰이는 모델로 변경함
        - Efficient net V2 large
    - 최종적으로는 mobile vision transformer - small로 모델 변경
        - 모델 앙상블을 적용하기 위해 단기간에 여러 모델을 훈련해야 했는데, CNN 계열 모델보다 훨씬 빠른 연산이 가능하기 때문 (파라미터 개수 5.3M)
        - gpu 1개만으로 32개 배치를 처리할 수 있었음
            - efficient net의 경우 gpu 2개 필요
- Text
    - 초기에는 단순한 Linear Feed forward network를 이용함
    - 이후 klue/roberta-large (huggingface transformer) 모델을 이용
        - 마지막 output layer의 cls 토큰에 linear classifier (1024 : 임베딩 크기 → 128 : cat3 개수)를 추가

### Loss function, 학습 방법, 그리고 모델 구조

- (1) Hierarchical Loss Network
    - train data에서 cat1, cat2가 주어진다는 점을 이용하고 싶었음
        - [https://github.com/Ugenteraan/Deep_Hierarchical_Classification](https://github.com/Ugenteraan/Deep_Hierarchical_Classification)
        - [https://arxiv.org/pdf/2005.06692.pdf](https://arxiv.org/pdf/2005.06692.pdf)
        - Deep Hierarchical Classification
    - 모델이 cat1을 예측한 값을 “모델 내에서” cat2의 예측 과정에 이용할 수 있도록 구현
    - (cat1 예측 값의 도움을 받아 계산된) cat2에 대한 예측 값을 cat3의 예측 과정에 이용할 수 있도록 구현
    - dependence loss를 활용하여 Hierarchical Loss를 계산할 수 있도록 구현
        - layer loss
        - dependence loss
- (2) 단순히 cat1에 대한 loss, cat2에 대한 loss, cat3에 대한 loss를 더하는 방법도 있었음
    - 일반적인 CrossEntropyLoss를 이용
    - ex. total_loss = (0.05 * cat1_loss) + (0.15* cat2_loss) + (0.85 * cat3_loss)
- (2)의 방법이 성능이 더 좋아서 이를 이용함
- 최종적으로는 epoch에 따라 cat1, cat2, cat3에 의해 발생되는 loss의 비율을 조정한 방식을 이용하였음
    - 0~10 epoch : 
    total_loss **=** loss1 x **0.05** **+** loss2 x **0.1** **+** loss3 x **0.85**
    - 10~15 epoch : 
    total_loss **=** loss1 x **0.025** **+** loss2 x **0.05** **+** loss3 x **0.925**
    - 15~ epoch:
    total_loss **=** loss1 x **0.01** **+** loss2 x **0.03** **+** loss3 x **0.96**
- Loss의 계산식은 변경하였지만, Hierarchical Loss Network 적용 과정에서 구현되었던 모델 구조는 그대로 이용함
    - cat1을 예측하기 위해 사용된 결과 값이 cat2 예측에 흘러들어감          
    - cat2를 예측하기 위해 사용된 결과 값이 cat3 예측에 흘러들어감
    - linear classifier + concat
- 대회 막바지에 이르러 모델 앙상블을 도입하였음
    - 동일한 모델을 random seed만 바꿔서 새로 train 시키고, 각각의 모델이 뽑아내는 예측 값을 soft voting ( → softmax 결과 값을 summation) 하여 inference 하였음
    - 총 10개의 모델을 만들었으나, 가장 높은 score를 기록한 것은 모델을 4개만 골랐을 때였음

### 4. Self-feedback?

### 의의 :

- Multimodality 에 대해 이해할 수 있었음
    - 단순히 두 modality에 대한 vector를 concat하는 것보다 더 좋은 방법은 없을까…? 에 대한 답을 얻지 못해 아쉽다.  
- CV 관련 모델을 두루두루 배울 수 있었음
    - inception network
        - 구닥다리 모델이긴 하지만, 다양한 크기의 필터를 이용하는 아이디어는 가져갈만 한 것 같다.
    - efficient net, mobile ViT
        - 앞으로 CV관련 baseline은 무조건 이 두 모델을 이용하게 될 것 같다.
        - 특히 mViT의 경우 파라미터가 5M 밖에 안 되는데도 성능이 상당히 좋아 더욱 애용할 것 같다.
- data preprocessing, data augmentation 연습
    - data preprocessing을 모듈화하는 것이 효율적임
    - augmented data를 어떻게 보관해야 하는지에 대한 idea  → pickle 혹은 h5py (다음 대회에 이용)
    - data augmentation 과정에서 발생하는 “복제품” 문제를 해결할 idea → iter마다 갱신되는 random seed
- 모델은 앙상블하는 것이 유리하다
    - 앙상블만으로 f1 score 82 → f1 score 84 로 2점을 챙길 수 있었음

### 개선할 점 :

1. 데이터 관련 : **Mislabeled data!**
    - 잘못 label 된 데이터가 얼마나 많은지 확인할 것.  
    - 가능하다면 mislabel을 handling 하는 데 최대한 많은 시간을 써야 한다.  
    - 테스트 데이터에 비슷한 noise가 있더라도 학습에 이용되는 데이터는 최대한 깨끗해야 한다.
    
2. 모델 관련 : 
    - **SOTA 모델을 baseline**으로 잡을 것
        - 최소한 널리 이용되는 baseline 모델을 이용할 것
    - customize가 하고 싶다면 SOTA 모델 위에다 할 것 !!
        - 시간낭비 금지!
        
3. 코드 구조 관련:
    - 조금 더 간결한 모듈화 필요

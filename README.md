# 듀오링고를 따라 만들어본 단어 맞추기 게임입니다

스트림릿을 통해서 구현하는 과정에서 hugging face와 스트림릿 사이의 연결 충돌이 생겨 문제를 해결 중입니다.


##### 사용한 모델 및 데이터셋
- 영어 문잘 생성 : https://huggingface.co/google/flan-t5-small
- 영어 한글 번역 : https://huggingface.co/facebook/m2m100_418M
- 사용 단어 데이터 셋 : https://www.kaggle.com/datasets/anneloes/wordgame?resource=download&select=Concreteness_english.csv

### 수정
스트림릿의 작동이 원활하지 않아 급하게 주피터 노트북 환경에서도 작동하게 변경했습니다.

##### 작동 예시

- 문제
<img width="674" height="180" alt="image" src="https://github.com/user-attachments/assets/dd4da280-c50c-477c-b86b-985b317e8fbc" />

- 정답
<img width="655" height="215" alt="image" src="https://github.com/user-attachments/assets/17cec56d-fc8e-4268-af76-c1c0e50ec110" />

- 오답
<img width="667" height="216" alt="image" src="https://github.com/user-attachments/assets/e7dbe74f-9d5e-4c27-b712-23ed91335d94" />

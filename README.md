# Webtoon-To-Speech
시각 및 감정적 요소를 반영한 웹툰 음성 변환 <br>
📅 2024.09 ~ 2025.01
<br></br>
## 📖 Overview
- 현재 웹툰은 시각 중심의 콘텐츠로, 청각적 몰입이 부족함
- 성우 기반 오디오 웹툰은 높은 제작 비용으로 인해 대중적으로 활용하기 어려움
- 목표 : 웹툰의 **시각적 정보**와 대사의 **감정적 정보**를 결합해 더욱 풍부하고 실감 나는 청각 경험을 제공
---

## 🏗️ 전체 파이프라인
![Image](https://github.com/user-attachments/assets/bef66021-3bd7-4fc0-9906-1346f939948c)
<br></br>

---

## 🔥 주요 도전 과제 및 해결 방법
| 과제 |  설명 |
|--------|---------|
| **말풍선 형태 분석** | 일반 대화, 감정 표현, 내레이션 등 다양한 말풍선 유형을 인식해야 함 |
| **대화 순서 결정** | 웹툰의 다양한 말풍선 배치 방식에 따라 적절한 순서로 출력해야 함 |
| **화자 할당 문제** | Tail tip(말풍선 꼬리 방향) 및 문맥을 고려한 화자 매칭 필요 |
| **감정 반영 음성 합성** | 캐릭터의 성별, 나이, 감정에 따라 적절한 음성 톤을 생성해야 함 |

### 1️⃣ 말풍선 형태 분석   <br>
✔️ **말풍선 탐지**  
   - YOLOv8 Segmentation (Fine-tuned for speech bubbles)
   - MSER 기반 나레이션 감지 (Tail Tip이 없는 직사각형 박스 검출)  
✔️ **말풍선 분류**
   - ㄹㄹ
   - 
### 2️⃣대화 순서 결정
  - 위에서 아래,왼쪽에서 오른쪽 정렬 
### 3️⃣화자 할당
### 4️⃣ 캐릭터 식별
🛠️
<br></br>
### ✅ Speech Synthesis Solution
1️⃣ **감정 분석**
2️⃣ **음성 합성**


---
## 📌 결과 
| Task |  Accuracy |
|--------|---------|
| **Speech Balloon Detection** | 92.22% |
| **Speech Balloon Classification** | 96.39% |
| **Character Detection and Identification** | 72.31% |
| **Balloon-to-Character Association** |  84.75% |

- `Speech Balloon Classification` : boudning box의 크기를 20% 확장했을 때 기존 성능 대비 **44.58% 개선**
-  `Balloon-to-Character Association` :  baseline magiv2 모델 성능 대비 **20.34% 개선**

🔗음성 합성 결과 보러가기 [링크를 클릭해주세요!][]
---
<br></br>
## 🔧 Trouble Shooting 



## Team 

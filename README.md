
## **데이터셋 설명**  

- **CIFAR-100**: 100개의 클래스로 구성된 이미지 데이터셋입니다.  
- **5개의 태스크(Task)**: 각 태스크는 **20개 클래스**로 구성됩니다.  
- **학습 및 테스트 데이터셋 생성**: 각 태스크에 대해 별도의 학습 및 테스트 데이터셋을 생성합니다.  
- **데이터 불균형**: 일부 실험에서는 클래스별 불균형 비율을 적용하여 데이터를 변형했습니다.  

---

## **실험 설정**

1. **CAM 클래스 선택 전략 (`cam_strategy`)**  
   CAM 클래스 선택 시 사용된 전략:  
   - **`top`**: 특징 벡터의 Norm이 가장 큰 클래스 선택  
   - **`lowest`**: 특징 벡터의 Norm이 가장 작은 클래스 선택  
   - **`random`**: 무작위로 클래스 선택  
   - **`middle`**: Norm 값이 중간인 클래스 선택  

2. **CAM 클래스 비율 (`cam_percentages`)**  
   CAM 클래스의 선택 비율:  
   - **[25%, 50%, 75%, 100%]**  

3. **CAM 해상도 (`cam_resolutions`)**  
   CAM 맵의 해상도:  
   - **[(7, 7), (14, 14), (28, 28)]**  

4. **CAM 임계값 (`cam_thresholds`)**  
   CAM 맵 이진화에 사용된 임계값:  
   - **[None, 0.3, 0.5, 0.7]**  

5. **데이터 불균형 비율 (`imbalanced_ratios`)**  
   클래스 불균형 비율을 적용한 실험:  
   - **[0.1, 0.2, 0.5]**  

6. **모델 구조 (`model_variants`)**  
   사용된 두 가지 모델 구조:  
   - **`baseline`**: 기본 CNN 모델  
   - **`deeper_conv`**: 더 깊은 CNN 구조  

7. **테스트를 위한 미사용 클래스 (`unseen_class_sets`)**  
   - **[80-89번 클래스]**, **[90-99번 클래스]**  

8. **CAM 추출 방식 (`cam_extraction_method`)**  
   - **`gradcam`**  
   - **`gradcam++`**  
   - **`scorecam`**  

9. **학습 Epoch 수 (`num_epochs`)**  
   - **5 Epoch**로 고정  

---

## **실험 절차**

1. **Teacher 모델 학습**  
   - 각 태스크의 데이터셋을 사용하여 Teacher CNN 모델을 학습합니다.  
   - Feature Map으로부터 **CAM 맵**을 생성합니다.  

2. **Student 모델 학습**  
   - Teacher 모델이 생성한 **CAM 맵**을 이용해 Student 모델에 지식을 증류합니다.  
   - **교차 엔트로피 손실(Cross-Entropy Loss)**와 **CAM 증류 손실**을 사용합니다.  

3. **하이퍼파라미터 변경**  
   - **CAM 전략**, **클래스 비율**, **해상도**, **데이터 불균형 비율** 등 다양한 조합으로 실험을 진행합니다.  

4. **모델 평가**  
   - 학습된 Student 모델의 정확도를 다음 기준으로 평가합니다:  
     - **Seen Accuracy**: 학습된 클래스의 정확도  
     - **Unseen Accuracy**: 학습되지 않은 클래스의 정확도  

---

## **결과 기록**

- 각 실험은 다음 정보를 기록합니다:  
  - **`task_idx`**: 태스크 인덱스  
  - **`cam_strategy`**: CAM 전략  
  - **`cam_percentage`**: CAM 클래스 비율  
  - **`cam_resolution`**: CAM 해상도  
  - **`cam_threshold`**: CAM 임계값  
  - **`imbalanced_ratio`**: 데이터 불균형 비율  
  - **`model_variant`**: 모델 구조  
  - **`cam_extraction_method`**: CAM 추출 방법  
  - **`seen_acc`**: 학습된 클래스 정확도  
  - **`unseen_acc`**: 미사용 클래스 정확도
 
'cam_extraction_method','transfer_unseen_cam' 구현이 미흡해, 아직 반영하지않았습니다. 랩장님 컨험후 반영하도록 하겠습니다.

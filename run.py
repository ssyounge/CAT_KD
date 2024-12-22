import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import copy
import torch.nn.functional as F
from datetime import datetime
from math import ceil
import csv
import sys
from tqdm import tqdm


# Updated Experiment Configuration

experiment_config = {
    "cam_strategy": ["top", "lowest", "random", "middle"],
    "cam_percentages": [25, 50, 75, 100],
    "cam_resolutions": [(7, 7), (14, 14), (28, 28)],
    "cam_thresholds": [None, 0.3, 0.5, 0.7],
    "imbalanced_ratios": [0.1, 0.2, 0.5],
    "model_variants": ["baseline", "deeper_conv"],
    "unseen_class_sets": [list(range(80, 90)), list(range(90, 100))],
    "cam_extraction_method": ["gradcam", "gradcam++", "scorecam"],
    "num_epochs": 5
}


class TeeLogger:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)  
        self.file.write(message)      

    def flush(self):
        self.terminal.flush()
        self.file.flush()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset_cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset_cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

task_classes_cifar100 = [
    list(range(0, 20)),
    list(range(20, 40)),
    list(range(40, 60)),
    list(range(60, 80)),
    list(range(80, 100))
]

def get_task_dataset(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    subset = [(dataset[i][0], dataset[i][1]) for i in indices]
    images, labels = zip(*subset)
    dataset = TensorDataset(torch.stack(images), torch.tensor(labels))
    return dataset

def get_unseen_class_dataset(dataset, seen_classes):
    unseen_indices = [i for i, (_, label) in enumerate(dataset) if label not in seen_classes]
    if len(unseen_indices) == 0:
        return None
    unseen_subset = [(dataset[i][0], dataset[i][1]) for i in unseen_indices]
    images, labels = zip(*unseen_subset)
    return TensorDataset(torch.stack(images), torch.tensor(labels))

task_train_datasets_cifar100 = [get_task_dataset(train_dataset_cifar100, classes) for classes in task_classes_cifar100]
task_test_datasets_cifar100 = [get_task_dataset(test_dataset_cifar100, classes) for classes in task_classes_cifar100]

def apply_class_imbalance(dataset, imbalance_ratio=0.1, log_distribution=False):
    """
    데이터셋에 불균형 비율을 적용.

    Args:
        dataset (TensorDataset): 원본 데이터셋.
        imbalance_ratio (float): 각 클래스별 샘플이 유지될 확률.
        log_distribution (bool): 불균형 적용 후 샘플 분포를 로그로 출력할지 여부.

    Returns:
        TensorDataset: 불균형이 적용된 데이터셋.
    """
    class_counts = {}
    for _, label in dataset:
        class_counts[label.item()] = class_counts.get(label.item(), 0) + 1

    keep_indices = []
    for i, (_, label) in enumerate(dataset):
        if np.random.rand() < imbalance_ratio or class_counts[label.item()] == 1:
            keep_indices.append(i)

    images, labels = zip(*[dataset[i] for i in keep_indices])

    # 부족한 샘플 보완
    expected_samples = {cls: int(count * imbalance_ratio) for cls, count in class_counts.items()}
    new_class_counts = {}
    for label in labels:
        new_class_counts[label.item()] = new_class_counts.get(label.item(), 0) + 1

    for cls, expected_count in expected_samples.items():
        actual_count = new_class_counts.get(cls, 0)
        if actual_count < expected_count * 0.9:  # 부족한 샘플 확인
            deficit = expected_count - actual_count
            indices = [i for i, (_, label) in enumerate(dataset) if label == cls]
            additional_indices = np.random.choice(indices, size=deficit, replace=True)
            keep_indices.extend(additional_indices)  # 보완된 샘플 추가

    # **경고 메시지 제거**
    # print(f"Warning: Class {cls} has fewer samples than expected ({actual_count} vs {expected_count})")

    # 분포 로그 비활성화
    if log_distribution:
        final_class_counts = {}
        for label in labels:
            final_class_counts[label.item()] = final_class_counts.get(label.item(), 0) + 1
        # print("Class distribution after imbalance:")
        # for cls, count in sorted(final_class_counts.items()):
        #     print(f"Class {cls}: {count} samples")

    return TensorDataset(torch.stack(images), torch.tensor(labels))


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3,padding=1), 
            nn.ReLU() 
        )
        self.classifier = nn.Linear(64*8*8, num_classes) 

    def forward(self, x):
        feature_maps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, feature_maps

def extract_cam(feature_maps, class_weights, cam_res=None, cam_threshold=None):
    B, C, H, W = feature_maps.shape  # B=batch size, C=channels, H=height, W=width
    spatial_size = H * W

    # Classifier weight 차원 확인 및 수정
    class_weights_flat = class_weights[:, :C * spatial_size].view(-1, C, spatial_size)
    feature_maps_flat = feature_maps.view(B, C, spatial_size) 

    # CAM 계산
    cams = torch.einsum('bck,ock->bok', feature_maps_flat, class_weights_flat)
    cams = cams.view(B, -1, H, W)  

    # CAM 해상도 조정
    if cam_res is not None:
        cams = F.interpolate(cams, size=cam_res, mode='bilinear', align_corners=False)

    # Threshold 적용
    if cam_threshold is not None:
        cams = (cams > cam_threshold).float()

    return cams


def cam_distillation_loss(student_cam, teacher_cam):
    return F.mse_loss(student_cam, teacher_cam)

def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    if test_loader is None:
        print("Test loader is None, skipping evaluation.")
        return 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def train_teacher(model, train_loader, num_epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()


class CAMStrategyRegistry:
    """CAM 클래스 선택 전략 등록 및 호출 시스템."""
    def __init__(self):
        self.strategies = {
            "top": self._select_top,
            "lowest": self._select_lowest,
            "random": self._select_random,
            "middle": self._select_middle,
        }

    def register_strategy(self, name, strategy_function):
        """새로운 전략 등록."""
        if name in self.strategies:
            raise ValueError(f"Strategy '{name}' is already registered.")
        self.strategies[name] = strategy_function

    def select_classes(self, weights, strategy, percentage):
        """지정된 전략에 따라 클래스 선택."""
        if strategy not in self.strategies:
            raise ValueError(f"Strategy '{strategy}' is not recognized.")
        return self.strategies[strategy](weights, percentage)

    def _select_top(self, weights, percentage):
        num_classes = weights.shape[0]
        used_count = int(ceil(num_classes * (percentage / 100.0)))
        norms = weights.norm(dim=1)
        sorted_indices = torch.argsort(norms, descending=True)
        return sorted_indices[:used_count]

    def _select_lowest(self, weights, percentage):
        num_classes = weights.shape[0]
        used_count = int(ceil(num_classes * (percentage / 100.0)))
        norms = weights.norm(dim=1)
        sorted_indices = torch.argsort(norms)
        return sorted_indices[:used_count]

    def _select_random(self, weights, percentage):
        num_classes = weights.shape[0]
        used_count = int(ceil(num_classes * (percentage / 100.0)))
        return torch.randperm(num_classes)[:used_count]

    def _select_middle(self, weights, percentage):
        num_classes = weights.shape[0]
        used_count = int(ceil(num_classes * (percentage / 100.0)))
        norms = weights.norm(dim=1)
        sorted_indices = torch.argsort(norms)
        mid_start = num_classes // 2 - used_count // 2
        return sorted_indices[mid_start:mid_start + used_count]


# CAM 전략 사용 예제
cam_strategy_registry = CAMStrategyRegistry()

# 사용자 정의 전략 추가
def custom_high_variance_strategy(weights, percentage):
    num_classes = weights.shape[0]
    used_count = int(ceil(num_classes * (percentage / 100.0)))
    variances = weights.var(dim=1)  # 클래스별 가중치 분산 계산
    sorted_indices = torch.argsort(variances, descending=True)
    return sorted_indices[:used_count]

cam_strategy_registry.register_strategy("high_variance", custom_high_variance_strategy)

# 기존 함수 수정
def select_classes_based_on_strategy(weights, strategy, percentage):
    """CAM 클래스 선택 전략 호출."""
    return cam_strategy_registry.select_classes(weights, strategy, percentage)

def student_distillation(student_model, teacher_model, train_loader,
                         unseen_loader=None,
                         cam_strategy="top",
                         cam_percentage=100,
                         cam_resolution=None,
                         cam_threshold=None,
                         lambda_cam=0.5,
                         num_epochs=5,
                         lr=0.001):
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    student_model.train()
    teacher_model.eval()

    with torch.no_grad():
        all_teacher_weights = teacher_model.classifier.weight.clone()

    used_class_indices = select_classes_based_on_strategy(all_teacher_weights, cam_strategy, cam_percentage)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # 일반 train_loader 학습
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_logits, student_features = student_model(images)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images)

            cls_loss = nn.CrossEntropyLoss()(student_logits, labels)

            student_cam = extract_cam_with_method(
                student_features[-1],
                student_model.classifier.weight,
                cam_res=cam_resolution,
                cam_threshold=cam_threshold
            )
            teacher_cam = extract_cam_with_method(
                teacher_features[-1],
                teacher_model.classifier.weight,
                cam_res=cam_resolution,
                cam_threshold=cam_threshold
            )

            student_cam = student_cam[:, used_class_indices, :, :]
            teacher_cam = teacher_cam[:, used_class_indices, :, :]
            cam_loss = cam_distillation_loss(student_cam, teacher_cam)

            total_loss = cls_loss + lambda_cam * cam_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        # Unseen CAM 데이터 학습
        if unseen_loader:
            for unseen_images, unseen_cams in unseen_loader:
                unseen_images, unseen_cams = unseen_images.to(device), unseen_cams.to(device)

                # 배치 차원 검증
                if unseen_images.size(0) != unseen_cams.size(0):
                    print(f"Skipping unseen CAM batch: {unseen_images.size()} vs {unseen_cams.size()}")
                    continue

                optimizer.zero_grad()
                student_logits, student_features = student_model(unseen_images)

                unseen_student_cam = extract_cam_with_method(
                    student_features[-1], student_model.classifier.weight,
                    cam_res=cam_resolution
                )
                unseen_cam_loss = cam_distillation_loss(unseen_student_cam, unseen_cams)
                unseen_cam_loss.backward()
                optimizer.step()
                epoch_loss += unseen_cam_loss.item()


        print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {epoch_loss:.4f}")

class CAMMethodRegistry:
    """CAM 메서드 등록 및 호출 시스템."""
    def __init__(self):
        self.methods = {
            "gradcam": self._extract_gradcam,
            "gradcam++": self._extract_gradcam_plus,
            "scorecam": self._extract_scorecam,
        }

    def register_method(self, name, method_function):
        """새로운 CAM 메서드 등록."""
        if name in self.methods:
            raise ValueError(f"Method '{name}' is already registered.")
        self.methods[name] = method_function

    def extract_cam(self, feature_maps, class_weights, method, **kwargs):
        """지정된 메서드를 사용해 CAM 생성."""
        if method not in self.methods:
            raise ValueError(f"Method '{method}' is not recognized.")
        return self.methods[method](feature_maps, class_weights, **kwargs)

    def _extract_gradcam(self, feature_maps, class_weights, **kwargs):
        return extract_cam(feature_maps, class_weights, **kwargs)

    def _extract_gradcam_plus(self, feature_maps, class_weights, **kwargs):
        return extract_gradcam_plus(feature_maps, class_weights, **kwargs)

    def _extract_scorecam(self, feature_maps, class_weights, **kwargs):
        images = kwargs.get("images", None)
        model = kwargs.get("model", None)
        if images is None or model is None:
            raise ValueError("Score-CAM requires 'images' and 'model' as arguments.")
        return extract_scorecam(feature_maps, class_weights, images, model, **kwargs)


# CAM 메서드 사용 예제
cam_method_registry = CAMMethodRegistry()

# 사용자 정의 CAM 메서드 추가 예제
def custom_thresholded_cam(feature_maps, class_weights, **kwargs):
    """간단한 사용자 정의 CAM 메서드."""
    cam = extract_cam(feature_maps, class_weights, **kwargs)
    threshold = kwargs.get("threshold", 0.5)
    return (cam > threshold).float()

cam_method_registry.register_method("thresholded_cam", custom_thresholded_cam)

# 기존 함수 수정
def extract_cam_with_method(feature_maps, class_weights, method="gradcam", **kwargs):
    """CAM 메서드 호출."""
    return cam_method_registry.extract_cam(feature_maps, class_weights, method, **kwargs)


def extract_gradcam_plus(feature_maps, class_weights, cam_res=None, cam_threshold=None):
    """
    Grad-CAM++ 방식으로 CAM 생성.

    Args:
        feature_maps (torch.Tensor): Feature map tensor (B, C, H, W).
        class_weights (torch.Tensor): Classifier weights (num_classes, feature_dim).
        cam_res (tuple): CAM 해상도.
        cam_threshold (float): Threshold for binarization.

    Returns:
        torch.Tensor: Grad-CAM++ tensor (B, num_classes, cam_res[0], cam_res[1]).
    """
    B, C, H, W = feature_maps.shape

    # Gradients 계산
    gradients = torch.autograd.grad(outputs=feature_maps.sum(), inputs=feature_maps,
                                     grad_outputs=torch.ones_like(feature_maps), create_graph=True)[0]

    # Grad-CAM++ 가중치 계산
    gradients_squared = gradients ** 2
    gradients_cubed = gradients ** 3
    alphas = gradients_squared / (2 * gradients_squared + gradients_cubed.sum(dim=(2, 3), keepdim=True) + 1e-6)

    weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)

    # 가중치를 활용하여 CAM 생성
    cam = torch.relu(torch.sum(weights * feature_maps, dim=1, keepdim=True))  # (B, 1, H, W)

    # 해상도 조정
    if cam_res:
        cam = F.interpolate(cam, size=cam_res, mode='bilinear', align_corners=False)

    # Threshold 적용
    if cam_threshold is not None:
        cam = (cam > cam_threshold).float()

    return cam


def extract_scorecam(feature_maps, class_weights, images, model, cam_res=None, cam_threshold=None):
    """
    Score-CAM 방식으로 CAM 생성.

    Args:
        feature_maps (torch.Tensor): Feature map tensor (B, C, H, W).
        class_weights (torch.Tensor): Classifier weights (num_classes, feature_dim).
        images (torch.Tensor): Input images (B, C, H, W).
        model (nn.Module): 모델 객체.
        cam_res (tuple): CAM 해상도.
        cam_threshold (float): Threshold for binarization.

    Returns:
        torch.Tensor: Score-CAM tensor (B, num_classes, cam_res[0], cam_res[1]).
    """
    B, C, H, W = feature_maps.shape
    cam = torch.zeros((B, C, H, W), device=feature_maps.device)  # 초기화

    with torch.no_grad():
        for b in range(B):
            input_image = images[b].unsqueeze(0)  # (1, C, H, W)
            for c in range(C):
                # 특정 채널의 Feature Map 활성화
                mask = feature_maps[b, c, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                mask = F.interpolate(mask, size=input_image.shape[2:], mode='bilinear', align_corners=False)

                # Mask를 입력에 곱함
                masked_image = input_image * mask
                logits, _ = model(masked_image)

                # Score 계산
                score = logits[:, c].item()  # 특정 클래스에 대한 Score
                cam[b, c, :, :] += score * feature_maps[b, c, :, :]  # 점수를 CAM에 적용

    cam = torch.relu(cam)  # ReLU 적용

    # 해상도 조정
    if cam_res:
        cam = F.interpolate(cam, size=cam_res, mode='bilinear', align_corners=False)

    # Threshold 적용
    if cam_threshold is not None:
        cam = (cam > cam_threshold).float()

    return cam

def student_distillation(
    student_model, teacher_model, train_loader,
    unseen_loader=None,  # unseen_loader를 활용할 수 있도록 유지
    cam_strategy="top", cam_percentage=100,
    transfer_unseen_cam=False,  # transfer_unseen_cam을 기본값 False로 유지
    unseen_classes=None, cam_resolution=None, cam_threshold=None,
    lambda_cam=0.5, num_epochs=5, lr=0.001
):
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    student_model.train()
    teacher_model.eval()

    with torch.no_grad():
        all_teacher_weights = teacher_model.classifier.weight.clone()

    # 클래스 선택 전략 적용
    used_class_indices = select_classes_based_on_strategy(all_teacher_weights, cam_strategy, cam_percentage)

    # Unseen class CAM 데이터 전이
    if transfer_unseen_cam and unseen_classes:  # transfer_unseen_cam 사용 시 처리
        unseen_cam_data = transfer_cam_to_unseen_classes(
            teacher_model, train_loader, unseen_classes, cam_resolution
        )
        if unseen_cam_data:  # 데이터가 생성된 경우 unseen_loader로 변환
            unseen_loader = DataLoader(unseen_cam_data, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # 에포크 당 손실값 저장

        # 일반 train_loader 학습
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Student와 Teacher 모델 출력 생성
            student_logits, student_features = student_model(images)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images)

            # CrossEntropy 손실 계산
            cls_loss = nn.CrossEntropyLoss()(student_logits, labels)

            # CAM Distillation 손실 계산
            student_cam = extract_cam_with_method(
                student_features[-1],
                student_model.classifier.weight,
                cam_res=cam_resolution,
                cam_threshold=cam_threshold
            )
            teacher_cam = extract_cam_with_method(
                teacher_features[-1],
                teacher_model.classifier.weight,
                cam_res=cam_resolution,
                cam_threshold=cam_threshold
            )

            num_classes = student_cam.shape[1]
            used_class_indices = used_class_indices[used_class_indices < num_classes]

            student_cam = student_cam[:, used_class_indices, :, :]
            teacher_cam = teacher_cam[:, used_class_indices, :, :]
            cam_loss = cam_distillation_loss(student_cam, teacher_cam)

            # 총 손실 계산 및 역전파
            total_loss = cls_loss + lambda_cam * cam_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Unseen Loader 학습
        if unseen_loader:
            for unseen_images, unseen_cams in unseen_loader:
                unseen_images, unseen_cams = unseen_images.to(device), unseen_cams.to(device)

                optimizer.zero_grad()

                student_logits, student_features = student_model(unseen_images)

                student_cam = extract_cam_with_method(
                    student_features[-1],
                    student_model.classifier.weight,
                    cam_res=cam_resolution
                )

                unseen_cam_loss = cam_distillation_loss(student_cam, unseen_cams)
                unseen_cam_loss.backward()
                optimizer.step()

                epoch_loss += unseen_cam_loss.item()

        # Epoch 종료 시 로그 출력
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Total Loss: {epoch_loss:.4f}")

    # Unseen CAM 데이터 전이 로그
    if transfer_unseen_cam and unseen_classes:
        print(f"Transferred {len(unseen_cam_data)} unseen class CAM samples.") if unseen_cam_data else None

def transfer_cam_to_unseen_classes(teacher_model, train_loader, unseen_classes, cam_res):
    """
    Teacher 모델을 이용해 unseen 클래스에 대한 CAM을 생성.

    Args:
        teacher_model (nn.Module): Teacher 모델.
        train_loader (DataLoader): 학습 데이터 로더.
        unseen_classes (list): Unseen 클래스 리스트.
        cam_res (tuple): CAM 해상도.

    Returns:
        DataLoader: Unseen 클래스에 대한 CAM 데이터 로더.
    """
    unseen_cam_data = []
    teacher_model.eval()

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits, feature_maps = teacher_model(images)
            
            for cls in unseen_classes:
                class_weights = teacher_model.classifier.weight[cls].unsqueeze(0)
                cam = extract_cam_with_method(
                    feature_maps[-1],
                    class_weights,
                    method="gradcam",  # or "gradcam++", "scorecam" based on config
                    cam_res=cam_res
                )
                unseen_cam_data.append((images.cpu(), cam.cpu()))  # CPU로 전환하여 저장

    if unseen_cam_data:
        unseen_images, unseen_cams = zip(*unseen_cam_data)
        unseen_dataset = TensorDataset(
            torch.cat(unseen_images, dim=0), 
            torch.cat(unseen_cams, dim=0)
        )
        unseen_loader = DataLoader(unseen_dataset, batch_size=64, shuffle=True)
        return unseen_loader

    return None


def log_data_loader_sample_sizes(data_loader, description="Data Loader"):
    """
    데이터 로더의 샘플 크기를 로그로 출력.

    Args:
        data_loader (DataLoader): 데이터 로더 객체.
        description (str): 데이터 로더 설명.
    """
    sample_sizes = []
    for i, (images, labels) in enumerate(data_loader):
        sample_sizes.append((images.size(), labels.size()))
        # 첫 번째 배치만 기록
        if i == 0:
            print(f"[{description}] Sample Size (First Batch):")
            print(f"  Images: {images.size()}")
            print(f"  Labels: {labels.size()}")
            break
    print(f"[{description}] Total Batches: {len(sample_sizes)}")


def summarize_model_performance(results, log_filename):
    """
    모델별 평균 성능을 요약하고 저장.

    Args:
        results (list): 실험 결과 리스트.
        log_filename (str): 로그 파일 경로.
    """
    import pandas as pd

    df_results = pd.DataFrame(results)

    # 모델별 평균 성능 계산
    avg_results = df_results.groupby("model_variant")[["seen_acc", "unseen_acc"]].mean()
    print("\nModel-wise Average Performance:")
    print(avg_results)

    # 로그 파일에 저장
    with open(log_filename, "a") as log_file:
        log_file.write("\nModel-wise Average Performance:\n")
        avg_results.to_string(log_file)

    # 모델별 성능 비교 데이터 반환
    return avg_results

def initialize_model(variant, num_classes):
    if variant == "baseline":
        return CNNModel(num_classes=num_classes).to(device)
    elif variant == "deeper_conv":
        return DeeperCNNModel(num_classes=num_classes).to(device)

def save_experiment_result(result, result_filename):
    with open(result_filename, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        writer.writerow(result)

def initialize_and_train_teacher(train_loader, num_classes, num_epochs=5, lr=0.001):
    """
    Teacher 모델을 초기화하고 학습하는 함수.

    Args:
        train_loader (DataLoader): 학습 데이터 로더.
        num_classes (int): 클래스 개수.
        num_epochs (int): 학습 에포크 수.
        lr (float): 학습률.

    Returns:
        nn.Module: 학습이 완료된 Teacher 모델.
    """
    teacher_model = CNNModel(num_classes=num_classes).to(device)
    train_teacher(teacher_model, train_loader, num_epochs=num_epochs, lr=lr)
    return teacher_model

def run_experiments(task_train_datasets, task_test_datasets, unseen_class_sets, num_tasks, num_classes, experiment_config):
    import itertools
    import pandas as pd

    # 파일 경로 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_logs_{timestamp}.txt"
    result_filename = f"experiment_results_{timestamp}.csv"

    # 로그 초기화
    log_file = open(log_filename, "w")
    print(f"Experiment started at {datetime.now()} | Logs: {log_filename} | Results: {result_filename}")

    # 실험 설정 파라미터 추출
    cam_strategies = experiment_config.get("cam_strategy", ["top"])
    cam_percentages = experiment_config.get("cam_percentages", [100])
    cam_resolutions = experiment_config.get("cam_resolutions", [None])
    cam_thresholds = experiment_config.get("cam_thresholds", [None])
    imbalanced_ratios = experiment_config.get("imbalanced_ratios", [0.1])
    model_variants = experiment_config.get("model_variants", ["baseline"])
    unseen_class_sets = experiment_config.get("unseen_class_sets", [list(range(80, 100))])
    cam_methods = experiment_config.get("cam_extraction_method", ["gradcam"])
    num_epochs = experiment_config.get("num_epochs", 5)
    lambda_cam = 0.5

    # 결과 저장 CSV 초기화
    fieldnames = [
        'task_idx', 'cam_strategy', 'cam_percentage', 'cam_resolution', 'cam_threshold',
        'imbalanced_ratio', 'model_variant', 'cam_extraction_method', 'seen_acc', 'unseen_acc'
    ]
    with open(result_filename, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # 모든 실험 조합 생성
    combinations = list(itertools.product(
        cam_strategies, cam_percentages, cam_resolutions, cam_thresholds,
        imbalanced_ratios, model_variants, unseen_class_sets, cam_methods
    ))
    print(f"Total experiments: {len(combinations)}")

    all_results = []  # 모든 결과 저장
    for experiment_idx, (cam_strat, cam_perc, cam_res, cam_thresh, imbalance_ratio, model_variant, unseen_classes, cam_method) in enumerate(
        tqdm(combinations, desc="Experiment Progress")
    ):
        # 간단한 실행 정보 출력
        print(
            f"Running experiment [{experiment_idx+1}/{len(combinations)}]: "
            f"cam_strategy={cam_strat}, cam_percentage={cam_perc}, cam_resolution={cam_res}, "
            f"cam_threshold={cam_thresh}, imbalance_ratio={imbalance_ratio}, model_variant={model_variant}, cam_method={cam_method}"
        )

        # 불균형 데이터셋 적용
        train_datasets = [
            apply_class_imbalance(ds, imbalance_ratio=imbalance_ratio, log_distribution=False)
            for ds in task_train_datasets
        ]

        for task_idx in range(num_tasks):
            train_loader = DataLoader(train_datasets[task_idx], batch_size=64, shuffle=True)
            test_loader = DataLoader(task_test_datasets[task_idx], batch_size=64, shuffle=False)
            unseen_ds = get_unseen_class_dataset(task_test_datasets[task_idx], unseen_classes)
            unseen_loader = DataLoader(unseen_ds, batch_size=64, shuffle=False) if unseen_ds else None

            # Teacher 모델 초기화 및 학습
            teacher_model = initialize_and_train_teacher(train_loader, num_classes, num_epochs=num_epochs, lr=0.001)

            # Unseen CAM 데이터 생성
            unseen_cam_loader = None
            if experiment_config.get("transfer_unseen_cam", False):
                unseen_cam_loader = transfer_cam_to_unseen_classes(
                    teacher_model=teacher_model,
                    train_loader=train_loader,
                    unseen_classes=unseen_classes,
                    cam_res=cam_res
                )

            # Student 모델 초기화
            student_model = initialize_model(model_variant, num_classes)

            # Student Distillation
            student_distillation(
                student_model, teacher_model, train_loader,
                unseen_loader=unseen_cam_loader,
                cam_strategy=cam_strat, cam_percentage=cam_perc,
                cam_resolution=cam_res, cam_threshold=cam_thresh,
                lambda_cam=lambda_cam, num_epochs=num_epochs
            )

            # 결과 평가
            seen_acc = evaluate_model(student_model, test_loader)
            unseen_acc = evaluate_model(student_model, unseen_loader) if unseen_loader else 0

            # 결과 저장
            result = {
                'task_idx': task_idx, 'cam_strategy': cam_strat, 'cam_percentage': cam_perc,
                'cam_resolution': cam_res, 'cam_threshold': cam_thresh, 'imbalanced_ratio': imbalance_ratio,
                'model_variant': model_variant, 'cam_extraction_method': cam_method,
                'seen_acc': seen_acc, 'unseen_acc': unseen_acc
            }
            all_results.append(result)
            save_experiment_result(result, result_filename)

            # 간단한 결과 출력
            print(
                f"Task {task_idx} | Model: {model_variant} | Seen Acc: {seen_acc:.2f}% | Unseen Acc: {unseen_acc:.2f}%"
            )

    # 전체 결과 요약 및 출력
    summarize_model_performance(all_results, log_filename)
    print(f"Experiment completed at {datetime.now()}")

    return all_results

# Deeper CNN 모델 예제 (더 깊은 네트워크)
class DeeperCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),  # 추가된 레이어
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 최종 출력: (B, 128, 4, 4)
        )
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature_maps.append(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits, feature_maps


# Log 파일 설정 및 실행
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"experiment_logs_{timestamp}.txt"

with open(log_filename, "w") as log_file:
    sys.stdout = TeeLogger(log_file)
    sys.stderr = TeeLogger(log_file)

    try:
        all_results = run_experiments(
            task_train_datasets_cifar100,
            task_test_datasets_cifar100,
            unseen_class_sets=experiment_config["unseen_class_sets"],
            num_tasks=len(task_classes_cifar100),
            num_classes=100,
            experiment_config=experiment_config,
        )
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

# 성능 요약
model_performance_summary = summarize_model_performance(all_results, log_filename)

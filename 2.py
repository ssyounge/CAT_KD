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

def apply_class_imbalance(dataset, imbalance_ratio=0.1):
    class_counts = {}
    for _, label in dataset:
        class_counts[label.item()] = class_counts.get(label.item(), 0) + 1
    keep_indices = []
    for i, (_, label) in enumerate(dataset):
        if np.random.rand() < imbalance_ratio or class_counts[label.item()] == 1:
            keep_indices.append(i)
    images, labels = zip(*[dataset[i] for i in keep_indices])
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

def select_classes_based_on_strategy(weights, cam_strategy, cam_percentage):
    """
    cam_strategy: "top", "lowest", "random", "middle"
    cam_percentage: e.g. 25, 50, 75, 100
    weights: classifier weight (num_classes, feature_dim)
    """
    num_classes = weights.shape[0]
    used_count = int(ceil(num_classes * (cam_percentage / 100.0)))
    
    # 간단히 각 클래스별 weight norm을 기준으로 선정
    norms = weights.norm(dim=1)  # 각 클래스 weight의 L2 norm
    sorted_indices = torch.argsort(norms) # norms 오름차순 정렬
    
    if cam_strategy == "top":
        # 가장 norm이 큰 클래스 선택(뒤에서 used_count개)
        selected = sorted_indices[-used_count:]
    elif cam_strategy == "lowest":
        # 가장 norm이 작은 클래스 선택(앞에서 used_count개)
        selected = sorted_indices[:used_count]
    elif cam_strategy == "random":
        # 랜덤 선택
        selected = torch.randperm(num_classes)[:used_count]
    elif cam_strategy == "middle":
        # 중간 부분 선택
        mid_start = num_classes//2 - used_count//2
        selected = sorted_indices[mid_start:mid_start+used_count]
    else:
        # 기본적으로는 top과 동일하게 처리
        selected = sorted_indices[-used_count:]
    
    return selected

def student_distillation(student_model, teacher_model, train_loader,
                         cam_strategy="top",
                         cam_percentage=100,
                         transfer_unseen_cam=False,
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

    # 클래스 선택 전략 적용
    used_class_indices = select_classes_based_on_strategy(all_teacher_weights, cam_strategy, cam_percentage)

    # unseen cam transfer 여부에 따른 추가 로직 가능
    # 여기서는 단순히 transfer_unseen_cam이 True일 때도 동일하게 used_class_indices 사용.
    # 실제론 seen/unseen을 구분하는 추가 로직 필요.

    for epoch in range(num_epochs):
        epoch_loss = 0.0  # 에포크 당 손실값 저장
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_logits, student_features = student_model(images)
            with torch.no_grad():
                teacher_logits, teacher_features = teacher_model(images)

            cls_loss = nn.CrossEntropyLoss()(student_logits, labels)

            # CAM Distillation
            student_cam = extract_cam(student_features[-1], student_model.classifier.weight,
                                    cam_res=cam_resolution, cam_threshold=cam_threshold)
            teacher_cam = extract_cam(teacher_features[-1], teacher_model.classifier.weight,
                                    cam_res=cam_resolution, cam_threshold=cam_threshold)

            # DEBUG 코드 추가
            DEBUG = False  # 디버깅을 활성화
            if DEBUG:
                print("num_classes in CAM:", student_cam.shape[1])
                print("used_class_indices:", used_class_indices)
                print("student_cam shape:", student_cam.shape)
                print("teacher_cam shape:", teacher_cam.shape)

            num_classes = student_cam.shape[1]
            used_class_indices = used_class_indices[used_class_indices < num_classes]

            student_cam = student_cam[:, used_class_indices, :, :]
            teacher_cam = teacher_cam[:, used_class_indices, :, :]
            cam_loss = cam_distillation_loss(student_cam, teacher_cam)

            total_loss = cls_loss + lambda_cam * cam_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()  # 에포크 총 손실값 업데이트
        
        # **에포크 단위 로그 출력**
        print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {epoch_loss:.4f}")

def run_experiments(task_train_datasets, task_test_datasets, unseen_class_sets, num_tasks, num_classes, experiment_config):
    import itertools

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_logs_{timestamp}.txt"
    result_filename = f"experiment_results_{timestamp}.csv"

    log_file = open(log_filename, "w")
    sys.stdout = TeeLogger(log_file)  
    sys.stderr = TeeLogger(log_file)

    print(f"Experiment started at {datetime.now()}")
    print("Saving logs to:", log_filename)
    print("Results will be saved to:", result_filename)

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

    # CSV 파일 초기화
    fieldnames = ['task_idx', 'cam_strategy', 'cam_percentage', 'cam_resolution', 
                  'cam_threshold', 'imbalanced_ratio', 'model_variant', 'cam_extraction_method',
                  'seen_acc', 'unseen_acc']

    # 중간 저장용 파일 생성
    with open(result_filename, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # 모든 실험 조합 생성
    combinations = list(itertools.product(
        cam_strategies, cam_percentages, cam_resolutions, cam_thresholds,
        imbalanced_ratios, model_variants, unseen_class_sets, cam_methods
    ))

    print(f"Total experiments: {len(combinations)}")
    for (cam_strat, cam_perc, cam_res, cam_thresh, imbalance_ratio, model_variant, unseen_classes, cam_method) in combinations:
        print(f"Running experiment: {cam_strat}, {cam_perc}%, {cam_res}, {cam_thresh}, {imbalance_ratio}, {model_variant}, {cam_method}")
        
        # 불균형 데이터 적용
        train_datasets = [apply_class_imbalance(ds, imbalance_ratio=imbalance_ratio) for ds in task_train_datasets]

        for task_idx in range(num_tasks):
            train_loader = DataLoader(train_datasets[task_idx], batch_size=64, shuffle=True)
            test_loader = DataLoader(task_test_datasets[task_idx], batch_size=64, shuffle=False)

            unseen_ds = get_unseen_class_dataset(task_test_datasets[task_idx], unseen_classes)
            unseen_loader = DataLoader(unseen_ds, batch_size=64, shuffle=False) if unseen_ds else None

            # Model 초기화
            student_model = CNNModel(num_classes=num_classes).to(device)
            teacher_model = CNNModel(num_classes=num_classes).to(device)

            # Teacher 학습
            train_teacher(teacher_model, train_loader, num_epochs=num_epochs, lr=0.001)
            base_student_acc = evaluate_model(student_model, test_loader)

            # Student Distillation
            student_distillation(student_model, teacher_model, train_loader,
                                 cam_strategy=cam_strat,
                                 cam_percentage=cam_perc,
                                 cam_resolution=cam_res,
                                 cam_threshold=cam_thresh,
                                 lambda_cam=lambda_cam,
                                 num_epochs=num_epochs)

            seen_acc = evaluate_model(student_model, test_loader)
            unseen_acc = evaluate_model(student_model, unseen_loader) if unseen_loader else 0

            # 결과 저장
            result = {
                'task_idx': task_idx,
                'cam_strategy': cam_strat,
                'cam_percentage': cam_perc,
                'cam_resolution': cam_res,
                'cam_threshold': cam_thresh,
                'imbalanced_ratio': imbalance_ratio,
                'model_variant': model_variant,
                'cam_extraction_method': cam_method,
                'seen_acc': seen_acc,
                'unseen_acc': unseen_acc
            }

            print(f"Task {task_idx} | Seen Acc: {seen_acc:.2f} | Unseen Acc: {unseen_acc:.2f}")
            
            # 중간 결과 저장
            with open(result_filename, "a", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)

    print(f"Experiment completed at {datetime.now()}")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()

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

run_experiments(task_train_datasets_cifar100, task_test_datasets_cifar100, 
                unseen_class_sets=experiment_config["unseen_class_sets"],
                num_tasks=len(task_classes_cifar100), num_classes=100, 
                experiment_config=experiment_config)

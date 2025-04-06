---
title: "TensorRT와 Triton: 딥러닝 추론 최적화의 강력한 조합"
date: 2024-07-30T15:30:00+09:00
lastmod: 2024-07-30T15:30:00+09:00
draft: false
description: "NVIDIA TensorRT와 Triton Inference Server의 내부 구조와 실전 최적화 경험을 공유합니다"
tags: [
	"deeplearning",
	"tensorrt",
  "nvidia triton",
  "inference optimization",
  "model deployment",
  "gpu acceleration"
]
categories: ["deeplearning"]
ShowToc: true
TocOpen: true
hidemeta: false
---

AI 모델을 배포해보려 공부해본 엔지니어라면 누구나 이런 고민을 해봤을 것이다. **"어떻게 하면 내 모델이 프로덕션 환경에서 더 빠르게 동작할 수 있을까?"**  <br>
처음 이런 task를 접했을 때는 단순 C/C++ 프로그래밍을 통해서 해결할 수 있을 줄 알았다. 그러나, 물론 naive한 python 보다야 낫지만, request가 많아질 수록 다른 해결책이 필요하다고 생각이 들 것이다.

그러던 와중에 나온 NVIDIA의 TensorRT 라는 놈이 있다. 오늘은 모델 최적화와 프로덕션 배포의 세계에서 강력한 도구로 자리잡은 TensorRT와 Triton Inference Server에 대해서 알아보자.

이 글에서는 단순한 "TensorRT 사용법"이나 "Triton 서버 설정법"이 아닌, 이 두 기술의 내부 구조, 최적화 원리, 그리고 실제 프로젝트에서 마주친 도전과 해결책에 대해 얘기해보고자 한다. <!--more-->

## TensorRT란?

TensorRT는 NVIDIA에서 개발한 고성능 딥러닝 추론 최적화 라이브러리로, 딥러닝 모델을 NVIDIA GPU에서 최대 성능으로 실행할 수 있도록 해준다. 이런 설명을 듣고 생각하면 일종의 CUDA 가속 추론 라이브러리 구나 생각이 들고, 으레 짐작할 수 있지만, TensorRT는 그 이상의 가치를 제공한다.

PyTorch나 TensorFlow에서 모델을 학습시킬 때, 이 프레임워크들은 유연성과 학습 편의성에 초점을 맞추고 있다. 그러나 프로덕션 환경에서는 순수한 추론 성능과 효율성이 훨씬 중요해진다. 여기서 TensorRT가 등장한다.

TensorRT의 핵심은 **계산 그래프 최적화(Computational Graph Optimization)**와 **하드웨어 최적화 커널(Hardware-Optimized Kernels)**이다. 여기서 부터 먼가 머리가 아파올 수 있다.. 계산 그래프를 최적화 하고 하드웨어 최적화 커널 이라니..? 무시무시한 최적화 방법인 것 같아 보인다. TensorRT는 모델의 구조를 분석하고 NVIDIA GPU에 최적화된 형태로 변환하여, 때로는 원본 모델보다 5-10배까지 빠른 추론 속도를 달성할 수 있다.

<!-- 검색 키워드: "TensorRT optimization workflow diagram", "ONNX to TensorRT conversion pipeline", "TensorRT layer fusion diagram"  
   이상적인 이미지는 원본 모델에서 ONNX 변환을 거쳐 TensorRT 엔진으로 최적화되는 전체 흐름과 레이어 융합, 메모리 최적화, 정밀도 조정 등의 핵심 기술이 시각화된 다이어그램입니다. -->
![TensorRT 최적화 파이프라인](/images/dl/tensorrt_optimization_flow.png)

## TensorRT의 마법: 내부 최적화 메커니즘

그렇다면 어떻게 TensorRT가 이렇게 극적인 성능 향상을 가능하게 하는지 조금 더 구체적으로 살펴보자.

### 1. 그래프 최적화 기법

TensorRT는 신경망 모델을 최적화하기 위해 여러 그래프 변환 기법을 사용한다:

- **레이어 융합(Layer Fusion)**: 여러 레이어를 단일 최적화된 레이어로 결합
- **커널 튜닝(Kernel Auto-Tuning)**: 하드웨어에 맞게 최적의 커널 구현 선택
- **텐서 메모리 최적화**: 메모리 사용량을 최소화하고 캐시 효율성 극대화
- **불필요한 연산 제거**: 추론 시 필요없는 계산 생략

내 경험상 **레이어 융합**은 가장 극적인 성능 개선을 가져오는 최적화 중 하나였다. 특히 Convolution + BatchNorm + ReLU와 같은 일반적인 패턴은 단일 최적화 커널로 대체되어 메모리 액세스를 크게 줄일 수 있었다.
나중에 OpenAI의 Triton에 대해서도 posting 하려고 하는데 여기서의 **메모리 액세스** 를 줄인다는게 최적화 과정의 아주 큰 요소이다. 이 메모리 액세스를 어떻게 줄인것인가에 많은 연구자들이 목을 메고 있다.

NVIDIA의 연구[7]에 따르면, 레이어 융합을 통해 중간 텐서의 메모리 액세스가 50% 이상 감소하고, 이는 대규모 모델에서 최대 25%의 추론 성능 향상으로 이어진다고 한다. Han 등의 연구[6]에서는 메모리 액세스 감소가 모바일 환경에서 에너지 효율성을 최대 3배까지 개선할 수 있음을 보여준다고 한다.

```cpp
// TensorRT의 레이어 융합 예시 (의사 코드)
// 이런 세 개의 레이어가 있다고 가정해보자
IConvolutionLayer* conv = network->addConvolution(...);
IBatchNormLayer* bn = network->addBatchNorm(...);
IActivationLayer* relu = network->addActivation(*bn->getOutput(0), ActivationType::kRELU);

// TensorRT는 내부적으로 이를 단일 최적화 커널로 융합한다
// network->addFusedConvBNRelu(...); (실제 API는 아님)
```

### 2. 정밀도 최적화(Precision Optimization)

TensorRT의 가장 강력한 기능 중 하나는 **혼합 정밀도 연산(Mixed Precision Computing)**이다. 다양한 수치 정밀도를 지원한다:

- **FP32**: 32비트 단정밀도 부동 소수점
- **FP16**: 16비트 반정밀도 부동 소수점 
- **INT8**: 8비트 정수 양자화
- **TF32**: NVIDIA Ampere GPU부터 지원하는 19비트 텐서 부동 소수점 포맷

실제 프로젝트에서, FP16으로 전환하는 것만으로도 **거의 정확도 손실 없이 2배 이상의 성능 향상**을 얻을 수 있었다. INT8로 더 나아가면 대략 4배까지 처리속도가 향상될 수 있지만, Quantization 과정에서 정확도 저하가 있기에 trade-off를 따져야 했다.

```python
# TensorRT의 정밀도 설정 예시
import tensorrt as trt

# builder 설정
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()

# FP16 정밀도 활성화
config.set_flag(trt.BuilderFlag.FP16)

# INT8 양자화 활성화 (보정 필요)
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = calibrator  # 보정기 인스턴스

# 엔진 생성
engine = builder.build_engine(network, config)
```

### 3. 동적 텐서 메모리와 실행 최적화

TensorRT는 실행 중에 텐서 메모리 할당을 최적화하기 위해 **동적 메모리 관리**를 사용한다. 이게 무슨 의미일까? 쉽게 설명하자면, 신경망이 연산을 수행할 때 필요한 메모리를 더 효율적으로 사용하는 방법들을 적용한다는 의미다. 여기에는 세 가지 핵심 전략이 있다.

#### 메모리 풀링(Memory Pooling)

기존 방식에서는 모델이 연산을 수행할 때마다 메모리를 할당하고 해제하는 과정을 반복했다. 이 과정은 GPU에게 "이 크기의 메모리가 필요해", "이제 이 메모리는 필요 없어" 라고 계속 말하는 것과 같은데, 이는 상당한 오버헤드를 발생시킨다.

메모리 풀링에서는 TensorRT가 미리 대량의 메모리 풀을 할당해두고, 필요할 때마다 이 풀에서 빠르게 메모리를 가져와 사용한다. 마치 공유 오피스에서 매번 책상을 사고 버리는 대신, 미리 준비된 책상을 필요할 때 사용하는 것과 같다.

```python
# 기존 방식 (의사 코드)
for each_operation in model:
    memory = allocate_new_memory(size_needed)  # 시간 소요
    perform_operation_using(memory)
    free_memory(memory)  # 또 시간 소요

# 메모리 풀링 방식 (의사 코드)
memory_pool = allocate_large_pool_once()  # 초기에 한 번만 비용 지불
for each_operation in model:
    memory = get_from_pool(memory_pool, size_needed)  # 매우 빠름
    perform_operation_using(memory)
    return_to_pool(memory)  # 해제가 아닌 반환 (빠름)
```

#### 텐서 재사용(Tensor Reuse)

딥러닝 모델 내부에서는 많은 중간 결과(텐서)가 생성된다. 이들은 보통 비슷한 크기를 가지는 경우가 많고, 한 번 사용된 후에는 더 이상 필요하지 않은 경우가 많다.

TensorRT는 똑똑하게도 "이 텐서의 데이터는 더 이상 필요 없으니, 이 메모리 공간을 다른 비슷한 크기의 텐서를 위해 재사용할 수 있겠군!"이라고 판단한다. 이는 특히 메모리가 제한적인 엣지 디바이스에서 중요하다.

예를 들어, ResNet과 같은 모델에서는 각 레이어의 출력 텐서가 비슷한 크기를 갖는다. 1번 레이어의 결과가 2번 레이어에 전달된 후에는 1번 레이어의 출력 메모리 공간을 3번 레이어의 출력을 저장하는 데 재사용할 수 있다.

```
[초기 상태]
Layer 1 출력 → 메모리 A 사용
Layer 2 출력 → 메모리 B 사용
Layer 3 출력 → 새 메모리 필요...

[텐서 재사용 적용]
Layer 1 출력 → 메모리 A 사용
Layer 2 출력 → 메모리 B 사용
Layer 3 출력 → 메모리 A 재사용 (Layer 1 출력은 이미 Layer 2에 전달됨)
```

#### 실행 병렬화(Execution Parallelism)


그래프 관점에서 보면, 딥러닝 모델은 일부 연산들이 서로 독립적이라 동시에 실행될 수 있다. TensorRT는 이러한 독립적 연산들을 찾아내 병렬로 실행한다.

예를 들어, 멀티 헤드 어텐션에서 각 "헤드"는 독립적으로 계산될 수 있다. 또한 모델의 여러 브랜치 (예: ResNet의 residual connection과 main path)도 병렬로 처리될 수 있다.

NVIDIA의 연구[7]에서는 다중 스트림 병렬화가 특히 Transformer 모델에서 효과적임을 보여주었다. 각 어텐션 헤드를 별도의 CUDA 스트림에 할당하는 기법만으로도 약 40%의 처리량 증가를 달성했다고 한다.

```cpp
// TensorRT의 작업 스케줄링 최적화 (의사 코드)
// 실제 구현은 더 복잡하지만, 개념적으로는 이런 방식

// 독립적인 두 연산 병렬 실행
// A와 B는 서로 의존성이 없다고 가정 (예: 두 개의 독립된 컨볼루션)
executeParallel([&]() {
  executeOperation(operationA);  // GPU의 일부 코어에서 실행
}, [&]() {
  executeOperation(operationB);  // 동시에 다른 코어에서 실행
});

// C는 A와 B의 결과에 의존 (예: A와 B의 결과를 합치는 연산)
// A와 B가 모두 완료될 때까지 기다린 후 실행
executeOperation(operationC);
```

이런 최적화 방법을 통해 지연 시간이 감소 되고, 그에 따라 처리량도 자연스레 증가 할 수 있을것이다.

## 실전 TensorRT 모델 변환과 최적화

실제 프로젝트에서 TensorRT를 사용할 때 알아야 할 중요한 워크플로우와 실무 팁을 살펴보자.

### 1. 모델 변환 파이프라인

TensorRT 모델 변환은 일반적으로 다음 단계를 따른다:

1. **원본 모델 내보내기**: PyTorch/TensorFlow 모델을 ONNX 또는 UFF 포맷으로 변환
2. **TensorRT 엔진 빌드**: ONNX/UFF 모델을 TensorRT 엔진으로 최적화
3. **정밀도 선택 및 보정**: 필요시 INT8 양자화를 위한 보정 수행
4. **엔진 직렬화(Serialization)**: 최적화된 엔진을 디스크에 저장하여 로딩 시간 단축

ONNX를 통한 변환이 가장 안정적이고 지원되는 경로였지만, PyTorch 모델에 따라 ONNX 변환 과정에서 여러 난관을 마주치기도 했다.

```python
# PyTorch에서 TensorRT 변환 예시
import torch
import tensorrt as trt

# 1. ONNX로 내보내기
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# 2. TensorRT 엔진 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# 3. 엔진 저장
with open("model.trt", "wb") as f:
    f.write(engine.serialize())
```

### 2. 동적 형상(Dynamic Shapes) 처리의 함정

TensorRT 7.0부터 동적 입력 크기가 잘 지원되지만, 최적의 성능을 얻기 위해서는 **최적화 프로필(Optimization Profiles)**을 신중하게 설정해야 한다.

NLP 모델을 배포했을 때 가변 길이 시퀀스를 효율적으로 처리하는 것이 큰 도전이었다. 예상 시퀀스 길이 범위에 맞게 최적화 프로필을 구성했을 때 가장 좋은 결과를 얻을 수 있었다.

```python
# 동적 형상을 위한 최적화 프로필 설정
config = builder.create_builder_config()
profile = builder.create_optimization_profile()

# 최소, 최적, 최대 입력 크기 지정
profile.set_shape(
    "input",                          # 입력 텐서 이름
    (1, 3, 224, 224),                 # 최소 크기
    (8, 3, 224, 224),                 # 최적 크기 (가장 자주 사용될 것으로 예상)
    (16, 3, 224, 224)                 # 최대 크기
)
config.add_optimization_profile(profile)
```

또한 다양한 배치 크기에 최적화된 **다중 프로필**을 사용하여 성능을 더욱 향상시킬 수 있었다.

### 3. INT8 양자화와 보정의 과학

FP16보다 더 극적인 성능 향상을 위해 INT8 양자화를 적용할 수 있지만, 이는 세심한 보정(Calibration) 과정이 필요하다:

1. **보정 데이터셋 준비**: 실제 데이터의 통계적 특성을 대표하는 샘플 준비
2. **보정기(Calibrator) 구현**: 다양한 보정 알고리즘 중 선택 (엔트로피, 최소-최대 등)
3. **보정 실행**: 보정 데이터를 통해 모델의 활성화 범위 분석
4. **정확도 검증**: 양자화 후 정확도 손실 평가

NVIDIA 개발자 블로그와 TVM 프로젝트 연구[8]에 따르면, 양자화 방식 중 특히 **엔트로피 보정** 방식이 가장 낮은 정보 손실로 양자화를 수행할 수 있어 정확도 측면에서 우수한 결과를 제공한다. 이 방법은 활성화 분포를 고려하여 양자화 스케일을 동적으로 조정하기 때문에 이미지 분류 모델에서 특히 효과적이다.

```python
# TensorRT INT8 보정 예시
import tensorrt as trt
import numpy as np

# 보정기 클래스 정의
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size, input_name):
        super().__init__()
        self.data = calibration_data
        self.batch_size = batch_size
        self.input_name = input_name
        self.current_idx = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)
        
    def get_batch_size(self):
        return self.batch_size
        
    def get_batch(self, names):
        if self.current_idx >= len(self.data):
            return None
            
        batch = self.data[self.current_idx:self.current_idx+self.batch_size]
        self.current_idx += self.batch_size
        
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [self.device_input]
        
    def read_calibration_cache(self):
        # 이전 보정 캐시 읽기 (있는 경우)
        return None
        
    def write_calibration_cache(self, cache):
        # 보정 결과 저장
        return None

# 보정기 사용
calibration_data = [...]  # 보정 데이터 준비
calibrator = EntropyCalibrator(calibration_data, batch_size=8, input_name="input")
config.int8_calibrator = calibrator
config.set_flag(trt.BuilderFlag.INT8)
```

실제로 대규모 모델의 경우, INT8 양자화를 통해 **최대 4배의 처리량 향상**을 경험했지만, 이는 약간의 정확도 희생을 동반했다. 트레이드오프가 용인되는 비즈니스 시나리오에서만 활용했다.

## NVIDIA Triton Inference Server: 확장 가능한 모델 서빙의 핵심

TensorRT가 단일 모델의 성능을 최적화하는 데 초점을 맞춘다면, Triton Inference Server는 **대규모 프로덕션 환경에서의 모델 서빙**을 처리한다. 

### 1. Triton의 아키텍처와 주요 기능

Triton Inference Server는 다양한 프레임워크와 하드웨어에서 추론 서비스를 제공하기 위한 컨테이너화된 솔루션이다. NVIDIA 개발자 컨퍼런스[7]에서 소개된 Triton의 주요 기능은 다음과 같다:

- **다중 프레임워크 지원**: TensorRT, ONNX Runtime, TensorFlow, PyTorch 등
- **동적 배치 처리**: 요청을 자동으로 배치화하여 처리량 최적화
- **동시 모델 실행**: 여러 모델을 동시에 서빙
- **모델 앙상블**: 복잡한 추론 파이프라인 구성
- **모델 버전 관리**: 무중단 업데이트 지원
- **하드웨어 리소스 관리**: GPU 메모리와 계산 자원 할당 최적화

<!-- 검색 키워드: "NVIDIA Triton Inference Server architecture", "Triton server multiple backends diagram", "Triton inference workflow"  
   이상적인 이미지는 Triton 서버의 구성 요소와 클라이언트-서버 통신 흐름, TensorRT/ONNX/TensorFlow 등 다양한 백엔드 지원을 보여주는 아키텍처 다이어그램입니다. -->
![Triton 서버 아키텍처](/images/dl/triton_architecture.png)

### 2. 모델 리포지토리 구성과 최적화

Triton은 **모델 리포지토리**라는 표준화된 디렉토리 구조를 통해 모델을 관리한다. 이 구조는 각 모델의 다양한 버전과 구성을 체계적으로 관리할 수 있게 해준다.

```
model_repository/
  ├── model1/
  │     ├── config.pbtxt        # 모델 구성 파일
  │     └── 1/                  # 버전 1
  │         └── model.plan      # TensorRT 엔진
  ├── model2/
  │     ├── config.pbtxt
  │     ├── 1/                  # 버전 1 
  │     │   └── model.onnx      # ONNX 모델
  │     └── 2/                  # 버전 2
  │         └── model.onnx      # 업데이트된 ONNX 모델
  └── ensemble_model/
        ├── config.pbtxt        # 앙상블 구성
        └── 1/                  # 버전 1
```


### 3. 동적 배치 처리와 수평적 확장

Triton의 강력한 기능 중 하나는 **동적 배치 처리(Dynamic Batching)**이다. 이는 개별 추론 요청을 자동으로 배치화하여 처리량을 극대화한다.

```
# config.pbtxt의 동적 배치 설정 예시
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 128
dynamic_batching {
  preferred_batch_size: [8, 16, 32, 64]
  max_queue_delay_microseconds: 5000
}
```

NVIDIA의 연구[7]와 실제 테스트에 따르면, 적절한 배치 크기와 대기 시간 설정은 GPU 활용도를 최대 95%까지 높일 수 있는 것으로 나타났다. 이 설정을 통해 Triton은 최대 5ms 동안 요청을 대기시키면서 선호하는 배치 크기로 그룹화하여 처리한다.

또한 Kubernetes와 같은 환경에서 Triton 서버를 **수평적으로 확장**하여 대규모 트래픽을 처리할 수 있었다:

```yaml
# Kubernetes 배포 예시 (간략화)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: triton-server
spec:
  replicas: 5  # 서버 인스턴스 수
  selector:
    matchLabels:
      app: triton
  template:
    metadata:
      labels:
        app: triton
    spec:
      containers:
      - name: triton-server
        image: nvcr.io/nvidia/tritonserver:22.07-py3
        args: ["tritonserver", "--model-repository=/models"]
        resources:
          limits:
            nvidia.com/gpu: 1  # 인스턴스당 GPU 수
        volumeMounts:
        - name: model-volume
          mountPath: /models
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

## 실전 TensorRT와 Triton 적용 사례와 도전

실제 프로젝트에서 TensorRT와 Triton을 적용하면서 마주친 다양한 도전과 해결책을 살펴보자.

### 1. 복잡한 아키텍처 모델의 변환 도전

최신 AI 연구에서 나온 복잡한 아키텍처의 모델을 TensorRT로 변환하는 과정은 종종 도전적이었다. 특히 **트랜스포머 기반 모델**의 경우, 동적 형상과 함께 변환하는 것이 까다로웠다.

예를 들어, BERT 모델을 TensorRT로 최적화할 때는 다음과 같은 접근 방식이 도움이 되었다:

1. **플러그인 활용**: NVIDIA의 TensorRT 플러그인 사용 (예: Multi-Head Attention)
2. **퓨전 허용 패턴**: 특정 연산 시퀀스를 TensorRT가 최적화할 수 있도록 모델 구조 조정
3. **최적의 ONNX opset 버전 선택**: 모델 구조에 맞는 적절한 ONNX opset 사용

```python
# BERT용 TensorRT 최적화 설정 예시
config = builder.create_builder_config()
config.max_workspace_size = 4 << 30  # 4GB

# 플러그인 활용
plugin_registry = trt.get_plugin_registry()
bert_plugin_creator = plugin_registry.get_plugin_creator("CustomBertPlugin", "1", "")
plugin = bert_plugin_creator.create_plugin("bert_plugin", plugin_params)

# 네트워크에 플러그인 레이어 추가
plugin_layer = network.add_plugin_v2([...], plugin)
```

### 2. 추론 지연 시간 vs 처리량 최적화

실시간 추론이 필요한 애플리케이션(예: 자율주행)과 대량 배치 처리가 필요한 애플리케이션(예: 오프라인 영상 분석) 사이에는 상충관계가 있다.

**지연 시간 최적화**를 위해서는:
- 작은 배치 크기 사용
- 스트림 병렬성 최소화
- 정확한 워크로드에 맞는 프로필 최적화

**처리량 최적화**를 위해서는:
- 큰 배치 크기 활용
- 다중 CUDA 스트림으로 병렬 처리
- 메모리 사용량과 계산 효율성 균형

서비스의 SLA(Service Level Agreement)에 따라 이 두 목표 사이의 적절한 균형점을 찾아야 했다.

```python
# Triton에서 지연 시간 vs 처리량 최적화 설정 예시
instance_group {
  count: 2  # GPU당 모델 인스턴스 수
  kind: KIND_GPU
  gpus: [0]
}

# 지연 시간 최적화
dynamic_batching {
  max_queue_delay_microseconds: 100  # 매우 짧은 대기 시간
}

# 처리량 최적화
dynamic_batching {
  preferred_batch_size: [64, 128]
  max_queue_delay_microseconds: 5000  # 더 긴 대기 시간 허용
}
```

### 3. 메모리 최적화와 다중 모델 배포

대규모 멀티 모델 시스템에서는 **GPU 메모리 관리**가 중요한 도전 과제였다. Triton에서는 다음과 같은 전략을 사용했다:

- **인스턴스 그룹 최적화**: GPU당 적절한 모델 인스턴스 수 설정
- **모델 로드 정책**: 필요시에만 메모리에 로드하는 정책 사용
- **우선순위 스케줄링**: 중요한 모델에 리소스 우선 할당

```
# config.pbtxt의 메모리 최적화 설정
instance_group {
  count: 1
  kind: KIND_GPU
  gpus: [0]
}

# 메모리 효율적인 로드 정책
model_transaction_policy {
  decoupled: false
}

dynamic_batching { ... }

# 필요시에만 모델 로드
model_control_mode: EXPLICIT
```

특히 여러 모델을 동시에 서빙해야 하는 경우, **모델별 리소스 할당**을 세심하게 조정하여 전체 시스템 성능을 최적화할 수 있었다.

## 실전 디버깅과 성능 분석 도구

TensorRT와 Triton을 효과적으로 활용하려면 강력한 디버깅 및 성능 분석 도구가 필수적이다.

### 1. NVIDIA Nsight Systems

[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)는 GPU 작업의 타임라인을 시각화하여 병목 현상을 식별하는 데 유용하다. 특히 호스트와 디바이스 간의 동기화 이슈나 커널 실행 패턴을 분석하는 데 탁월했다.

```bash
# Nsight Systems로 Triton 서버 프로파일링
nsys profile -t cuda,nvtx -o profile_report --capture-range=cudaProfilerApi \
    tritonserver --model-repository=/models
```

### 2. TensorRT trtexec 도구

`trtexec`는 TensorRT 엔진을 벤치마킹하고 분석하기 위한 명령줄 도구로, 다양한 설정에서 모델 성능을 빠르게 평가할 수 있다.

```bash
# trtexec을 사용한 성능 벤치마킹
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --verbose \
        --shapes=input:8x3x224x224 \
        --iterations=100 \
        --avgRuns=10
```

이 도구를 통해 다양한 배치 크기, 정밀도 및 최적화 설정에 따른 성능 차이를 체계적으로 측정할 수 있었다.

### 3. Triton 모델 분석기

Triton은 배포된 모델의 성능을 분석하기 위한 `perf_analyzer` 도구를 제공한다:

```bash
# Triton 모델 성능 분석
perf_analyzer -m resnet50 \
              -u localhost:8000 \
              -i grpc \
              --concurrency-range 1:16 \
              --shape input:3,224,224
```

이 도구를 통해 동시성 수준에 따른 지연 시간과 처리량을 측정하고, 최적의 서버 구성을 결정할 수 있었다.

## TensorRT와 Triton의 현실적인 한계와 대안

TensorRT와 Triton은 강력하지만, 몇 가지 제한 사항도 존재한다:

### 1. 모델 호환성 도전

모든 모델이 TensorRT로 완벽하게 변환되지는 않는다. 특히:

- **커스텀 연산자**: 표준 연산자가 아닌 경우 플러그인 개발 필요
- **동적 제어 흐름**: 조건문과 루프가 복잡한 모델의 경우 변환 어려움
- **최신 아키텍처**: 최신 연구 모델은 지원이 지연될 수 있음

이런 경우의 대안으로, 복잡한 부분만 PyTorch로 유지하고 나머지를 TensorRT로 최적화하는 **하이브리드 접근법**을 사용했다.

### 2. 개발 편의성 vs 최적화 수준

TensorRT는 학습 프레임워크에 비해 개발 편의성이 떨어진다. 모델 최적화에 소요되는 엔지니어링 시간과 얻을 수 있는 성능 향상 사이의 균형을 고려해야 한다.

간단한 경우, [PyTorch 2.0의 torch.compile()](https://pytorch.org/docs/stable/generated/torch.compile.html)과 같은 기능이 적절한 대안이 될 수 있었다.

### 3. 멀티 클라우드 및 이종 환경 지원

NVIDIA GPU에 최적화된 TensorRT는 다른 하드웨어 환경에서는 활용할 수 없다. 멀티 클라우드 전략이나 이종 하드웨어 환경을 고려한다면, [Apache TVM](https://tvm.apache.org/)이나 [ONNX Runtime](https://onnxruntime.ai/)과 같은 대안을 함께 검토해야 했다.


## 결론

TensorRT와 Triton Inference Server는 딥러닝 모델의 프로덕션 배포에 있어 강력한 도구이다. 적절한 최적화와 구성을 통해 극적인 성능 향상과 비용 절감을 달성할 수 있지만, 이는 깊은 이해와 체계적인 접근을 필요로 한다. 결국은 많이 사용해보고 공부해야 적절한 최적화를 감당할 수 있을 것이라는 생각이 들었다.

앞으로도 이런 최적화 방법론들은 계속 발전할 것이며, 엔지니어로서 이러한 도구의 내부 작동 방식을 이해하는 것은 효과적인 AI 시스템 구축에 있어 도움이 되지 않을까 싶다.

## References

* [1] NVIDIA TensorRT 공식 문서: [https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [2] NVIDIA Triton Inference Server: [https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)
* [3] TensorRT 최적화 기술: [https://on-demand.gputechconf.com/gtc/2019/presentation/s9431-tensorrt-inference-optimization.pdf](https://on-demand.gputechconf.com/gtc/2019/presentation/s9431-tensorrt-inference-optimization.pdf)
* [4] Triton 성능 모범 사례: [https://github.com/triton-inference-server/server/blob/main/docs/user_guide/performance_tuning.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/performance_tuning.md)
* [5] NVIDIA Deep Learning Examples: [https://github.com/NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)
* [6] Han, S., Shen, H., Philipose, M., Agarwal, S., Wolman, A., & Krishnamurthy, A. (2016). MCDNN: An approximation-based execution framework for deep stream processing under resource constraints. In Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services (pp. 123-136).
* [7] Nvidia. (2022). TensorRT: A platform for deep learning inference. In GTC 2022.
* [8] Chen, T., Moreau, T., Jiang, Z., Zheng, L., Yan, E., Shen, H., ... & Ren, M. (2018). TVM: An automated end-to-end optimizing compiler for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (pp. 578-594). 
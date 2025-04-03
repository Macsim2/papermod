---
title: "ONNX Graph의 내부 구조와 최적화 여정"
date: 2024-07-28T15:30:00+09:00
lastmod: 2024-07-28T15:30:00+09:00
draft: false
description: "ONNX Graph의 내부 구조를 파헤치고 실전 최적화 경험을 공유합니다"
tags: [
	"deeplearning",
	"onnx",
  "model optimization",
  "inference",
  "computational graph",
  "model deployment"
]
categories: ["deeplearning"]
ShowToc: true
TocOpen: true
hidemeta: false
---

딥러닝 모델을 프로덕션에 배포해본 개발자라면 누구나 이런 의문을 품어봤을 것이다. **"내 모델이 ONNX로 변환되면 내부적으로 어떻게 표현되고 실행될까?"** 🤔 <br>
모델 최적화와 프로덕션 배포의 세계에서 ONNX(Open Neural Network Exchange)(오닉스라고 읽더라)는 대중적인 도구가 되었지만, 그 내부 구조와 최적화 메커니즘에 대해 깊이 이해하는 것은 어렵기도 하고, 그런 개발자가 많지도 않아보인다.

이 글에서는 단순히 "ONNX 변환 방법"이 아닌, ONNX Graph의 내부 구조, 최적화 과정, 그리고 실제 프로젝트에서 마주친 도전과 해결책에 대해 파헤쳐보려 한다. <!--more-->

## ONNX란?

ONNX(Open Neural Network Exchange)는 2017년 Facebook과 Microsoft가 공동으로 발표한 오픈 포맷으로, 다양한 머신러닝 프레임워크 간의 모델 교환을 가능하게 하는 표준이다. 하지만 ONNX는 단순한 "파일 포맷" 이상의 의미를 가진다.

여러분이 PyTorch나 TensorFlow에서 모델을 만들 때, 그 모델은 해당 프레임워크의 특정 추상화와 인터페이스를 통해 표현된다. 그러나 프로덕션 환경에서는 최적의 성능과 호환성을 위해 이 모델을 "중립적인" 형태로 표현할 필요가 있다. 여기서 ONNX가 등장한다.<br>
"중립적"이라는 표현도 좀 추상적이긴 하다. 나중에 설명할테니 여기선 "표준적"인 의미로 알고 넘어가도 괜찮을거 같다.

ONNX의 핵심은 **계산 그래프(Computational Graph)** 이다. 이 그래프는 모델의 연산과 데이터 흐름을 노드와 엣지로 표현한다. 각 노드는 특정 연산(Conv, MatMul, Add 등)을 나타내며, 엣지는 텐서 데이터의 흐름을 나타낸다.

![ONNX Graph의 개념적 표현](/images/dl/onnx_graph_concept.png)

## ONNX Graph의 해부학

ONNX 모델 파일을 열어보면, 그 내부에는 복잡한 계산 그래프가 Proto 버퍼 형식으로 저장되어 있다. 이 그래프의 기본 구성 요소를 살펴보자:

### 1. 노드(Node)와 연산자(Operator)

ONNX 그래프의 핵심 구성 요소는 **노드**다. 각 노드는 다음과 같은 속성을 가진다:

- **op_type**: 노드가 수행하는 연산의 유형 (예: Conv, MatMul, Relu)
- **inputs**: 입력 텐서의 이름 목록
- **outputs**: 출력 텐서의 이름 목록
- **attributes**: 연산에 필요한 매개변수 (예: 커널 크기, 스트라이드 등)

```python
# 간단한 ONNX 노드 예시
node {
  input: ["X", "W"]
  output: ["Y"]
  name: "conv1"
  op_type: "Conv"
  attribute {
    name: "kernel_shape"
    ints: 3, 3
    type: INTS
  }
  attribute {
    name: "strides"
    ints: 1, 1
    type: INTS
  }
}
```

ONNX를 사용하며 내가 마주한 사실중에 하나는, 동일한 PyTorch 모델도 ONNX로 변환하는 방식과 옵션에 따라 생성되는 노드의 종류와 수가 크게 달라질 수 있다는 점이었다. 특히 `torch.onnx.export()` 함수의 `opset_version` 매개변수는 그래프 구조에 상당한 영향을 미친다.

### 2. 텐서(Tensor)와 값(Value)

ONNX 그래프에서 노드 간에 흐르는 데이터는 **텐서**로 표현된다. 이 텐서들은 다음과 같은 속성을 가진다:

- **이름**: 텐서의 고유 식별자
- **데이터 타입**: 텐서의 요소 타입 (float32, int64 등)
- **형상(Shape)**: 텐서의 차원과 크기

특히 주목할 점은 ONNX에서 모델 가중치도 그래프의 일부로 직접 저장된다는 것이다. 이는 PyTorch의 `state_dict`나 TensorFlow의 체크포인트와는 다른 접근 방식이다.

```python
# ONNX 모델에서 가중치 텐서 예시
initializer {
  dims: 64
  dims: 3
  dims: 3
  dims: 3
  data_type: FLOAT
  name: "conv1.weight"
  raw_data: "..." # 실제 바이너리 데이터는 여기에 저장
}
```

### 3. 그래프 입력과 출력

ONNX 그래프는 명시적인 **입력**과 **출력** 정의를 가진다:

```python
# 그래프 입력 예시
input {
  name: "input"
  type {
    tensor_type {
      elem_type: FLOAT
      shape {
        dim {
          dim_value: 1  # 배치 크기
        }
        dim {
          dim_value: 3  # 채널
        }
        dim {
          dim_value: 224  # 높이
        }
        dim {
          dim_value: 224  # 너비
        }
      }
    }
  }
}
```

이 명시적인 입출력 정의는 모델 배포 시 큰 장점이 된다. 모델이 어떤 형태의 입력을 기대하고 어떤 형태의 출력을 생성할지 명확히 알 수 있기 때문이다.

## ONNX Graph 생성의 비밀

PyTorch나 TensorFlow 모델을 ONNX로 변환할 때, 단순히 "내보내기" 버튼을 누르는 것처럼 보일 수 있지만, 내부적으로는 복잡한 변환 과정이 일어난다.

### 1. 추적(Tracing)과 스크립팅(Scripting)

PyTorch에서 ONNX 변환은 주로 두 가지 방식으로 이루어진다:

- **추적(Tracing)**: 모델에 실제 입력 데이터를 통과시키면서 실행되는 연산을 기록
- **스크립팅(Scripting)**: 모델 코드를 분석하여 정적 그래프를 생성

이 두 방식은 각각 장단점이 있는데, 내 경험상 **동적 제어 흐름(if문, 루프 등)이 포함된 모델에서는 trace 방식이 문제를 일으키는 경우가 많았다**. 특히 입력 크기에 따라 동작이 달라지는 모델에서는 스크립팅 방식이 더 안정적인 결과를 제공했다.

```python
# 추적 방식 예시
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx", 
                 export_params=True,
                 opset_version=12,
                 do_constant_folding=True)

# 스크립팅 방식 예시 (TorchScript 이용)
scripted_model = torch.jit.script(model)
torch.onnx.export(scripted_model, dummy_input, "model_scripted.onnx")
```

### 2. 연산자 매핑의 함정

PyTorch나 TensorFlow의 연산자가 ONNX로 매핑되는 과정에서 많은 함정이 있다. 특히 커스텀 연산자나 최신 연산자의 경우 직접적인 ONNX 대응이 없을 수 있다.

내가 경험한 가장 까다로운 사례는 **attention 메커니즘**이 포함된 트랜스포머 모델의 변환이었다. <br>
PyTorch의 `MultiheadAttention` 모듈이 ONNX로 변환될 때, 단일 노드가 아닌 여러 기본 연산자(MatMul, Add, Softmax 등)의 복잡한 조합으로 분해되었다. 이로 인해 그래프가 매우 복잡해지고 최적화 기회가 제한되었다.

```python
# PyTorch의 MultiheadAttention이 ONNX로 변환되면
# 아래와 같은 여러 노드로 분해된다 (간략화된 버전)
node {
  op_type: "MatMul"
  input: ["queries", "key_weights"]
  output: ["QK"]
}
node {
  op_type: "Transpose"
  input: ["QK"]
  output: ["QK_transposed"]
}
node {
  op_type: "Div"
  input: ["QK_transposed", "scaling_factor"]
  output: ["scaled_QK"]
}
node {
  op_type: "Softmax"
  input: ["scaled_QK"]
  output: ["attention_weights"]
}
# ... 등등
```

ONNX opset 버전에 따라 이러한 패턴이 크게 달라질 수 있으며, 최신 opset에서는 더 효율적인 변환이 이루어지는 경우가 많다. 이는 ONNX 생태계가 지속적으로 발전하고 있음을 보여준다.

## ONNX Graph 최적화의 기술

ONNX 모델을 얻은 후에는 추가적인 최적화를 통해 더 나은 성능을 얻을 수 있다. 여기서 ONNX Runtime과 같은 추론 엔진이 핵심 역할을 한다.

### 1. 그래프 변환 최적화

ONNX Runtime은 그래프에 다양한 변환을 적용하여 최적화한다:

- **상수 폴딩(Constant Folding)**: 상수 입력만 가지는 노드의 결과를 미리 계산
- **노드 융합(Node Fusion)**: 특정 패턴의 노드들을 단일 최적화된 노드로 결합
- **불필요한 노드 제거**: 출력에 영향을 미치지 않는 연산 제거

이중 **노드 융합**은 가장 큰 성능 향상을 가져오는 최적화 중 하나이다. 예를 들어, Conv+BatchNorm+ReLU 패턴은 단일 융합 노드로 대체될 수 있어 메모리 접근과 연산을 크게 줄일 수 있다.


```python
# ONNX Runtime의 그래프 최적화 예시
import onnxruntime as ort

# 기본 세션 vs 최적화된 세션
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
optimized_session = ort.InferenceSession("model.onnx", sess_options)

# 최적화 전후 그래프 비교
baseline_session = ort.InferenceSession("model.onnx", 
                                      ort.SessionOptions())
print(f"원본 그래프 노드 수: {len(baseline_session._model_meta.custom_metadata_map)}")
print(f"최적화 그래프 노드 수: {len(optimized_session._model_meta.custom_metadata_map)}")
```

### 2. 양자화(Quantization)의 비밀

ONNX 모델의 또 다른 강력한 최적화는 **양자화**이다. FP32 정밀도의 모델을 INT8이나 심지어 INT4로 변환하여 메모리 사용량과 계산 비용을 크게 줄일 수 있다.

그러나 양자화에는 함정이 있다. 내 경험에서, 단순히 "양자화 버튼"을 누르는 것 같은 사후 훈련 양자화(Post-Training Quantization, PTQ)는 때때로 정확도를 크게 저하시킬 수 있다. 특히 **작은 모델이나 희소한 활성화 함수를 가진 모델**에서 이 문제가 두드러졌다.

이런 경우, 양자화 인식 훈련(Quantization-Aware Training, QAT)이나 보정 데이터를 사용한 세심한 PTQ가 더 나은 결과를 제공했다.

```python
# ONNX 모델의 양자화 예시 (간략화됨)
from onnxruntime.quantization import quantize_dynamic

# 동적 양자화
quantize_dynamic("model.onnx", "model_quantized.onnx",
                weight_type=QuantType.QInt8)

# 보정 데이터를 사용한 정적 양자화
from onnxruntime.quantization import quantize_static, CalibrationDataReader

calibration_data = CalibrationDataReader(...)  # 보정 데이터 로더
quantize_static("model.onnx", "model_quantized.onnx",
               calibration_data)
```

### 3. 하드웨어 특화 최적화

ONNX의 또 다른 강점은 다양한 하드웨어 백엔드에 대한 최적화가 가능하다는 점이다. 특히 ONNX Runtime은 CPU, GPU, EdgeTPU, DSP 등 다양한 하드웨어에 대한 실행 제공자(Execution Provider)를 지원한다.

내 프로젝트에서는 동일한 ONNX 모델을 CPU, CUDA, TensorRT 제공자로 실행했을 때 성능 차이가 극적이었다:

```
| 실행 환경                  |  추론 시간 (ms) |
|--------------------------|---------------|
| CPU (OpenMP)             | 85.2          |
| CUDA                     | 27.3          |
| TensorRT (FP16)          | 12.1          |
| TensorRT (INT8 양자화)     | 5.8           |
```

특히 TensorRT 제공자는 그래프를 더욱 최적화하고 텐서 코어와 같은 하드웨어 가속기를 활용하여 놀라운 성능 향상을 제공했다.

## 실전 ONNX Graph 디버깅과 해결책

ONNX 모델을 프로덕션에 배포하면서 몇 가지 까다로운 문제를 마주쳤다. 이러한 경험은 ONNX Graph의 내부 구조를 더 깊이 이해하는 데 도움이 되었다.

### 1. 동적 입력 크기의 함정

CNN 기반 모델에서는 입력 크기가 고정되어 있는 경우가 많지만, NLP나 시계열 모델에서는 **가변 길이 입력**을 처리해야 하는 경우가 많다. ONNX에서 이를 처리하기 위해 **동적 축(Dynamic Axes)**을 지정할 수 있지만, 이로 인해 특정 최적화가 불가능해지는 부작용이 있었다.

예를 들어, TensorRT 변환 시 동적 축을 가진 모델은 일부 최적화를 적용할 수 없었고, 특정 차원에 대해 최적화된 프로필을 명시적으로 제공해야 했다.

```python
# 동적 축을 가진 ONNX 모델 내보내기
dynamic_axes = {
    'input': {0: 'batch_size', 2: 'seq_length'},
    'output': {0: 'batch_size', 1: 'seq_length'}
}
torch.onnx.export(model, dummy_input, "dynamic_model.onnx",
                 dynamic_axes=dynamic_axes)

# TensorRT에서는 최적화 프로필 지정이 필요
import tensorrt as trt
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 16), (8, 3, 64), (16, 3, 128))
config.add_optimization_profile(profile)
```

### 2. 커스텀 연산자의 도전

표준 연산자만으로는 구현하기 어려운 특별한 로직이 필요한 경우 혹은 성능 최적화가 필수적인 경우 **커스텀 ONNX 연산자**를 구현할 수 있다. 이는 복잡한 과정이지만, ONNX의 확장성을 보여주는 사례다.

특수한 비디오 처리 연산이 필요했다고 하면, 이를 위해 C++로 커스텀 ONNX 연산자를 구현하고 ONNX Runtime에 등록할 수 있다.

```cpp
// 커스텀 ONNX 연산자 구현 (C++)
struct CustomOp {
  static constexpr const char* OpName = "CustomVideoProcessor";
  
  static Status Compute(OpKernelContext* context) {
    // 구현 로직
    return Status::OK();
  }

  static ONNX_NAMESPACE::OpSchema GetOpSchema() {
    ONNX_NAMESPACE::OpSchema schema;
    schema.SetName(OpName);
    schema.SetDomain("MyDomain");
    schema.SetDoc("Custom video processing operation");
    schema.Input(0, "X", "Input tensor", "T");
    schema.Output(0, "Y", "Output tensor", "T");
    schema.TypeConstraint("T", {"tensor(float)"}, "Supported types");
    return schema;
  }
};

// 연산자 등록
ORT_REGISTER_CUSTOM_OP(CustomOp);
```

이 접근 방식의 장점은 특수 로직을 효율적으로 구현할 수 있다는 것이지만, 단점은 **이식성이 떨어진다**는 점이다. 커스텀 연산자를 사용하는 ONNX 모델은 해당 연산자가 등록된 환경에서만 실행할 수 있다.

### 3. 메모리 최적화의 미학


- **메모리 패턴 최적화**: 동일한 형상의 텐서를 재사용
- **외부 메모리 사용**: 모델 가중치를 외부 파일로 분리
- **실행 계획 최적화**: 최소 메모리 사용량으로 연산 순서 재배열

특히 엣지 디바이스에 배포할 때, 이러한 최적화가 모델 실행 가능성을 결정짓는 중요한 요소였다.

```python
# 메모리 최적화된 ONNX Runtime 세션
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = True
sess_options.enable_mem_reuse = True
sess_options.add_session_config_entry("session.save_model_format", "ORT")
sess_options.optimized_model_filepath = "optimized_model.ort"
session = ort.InferenceSession("model.onnx", sess_options)
```

## ONNX Graph 분석 도구의 세계

ONNX 모델을 더 깊이 이해하고 최적화하기 위한 다양한 도구가 있다:

### 1. Netron: 그래프 시각화의 황금 표준

[Netron](https://github.com/lutzroeder/netron)은 ONNX 그래프를 시각적으로 탐색할 수 있는 강력한 도구다. 각 노드의 속성, 입출력 텐서 형상, 가중치 분포까지 확인할 수 있어 디버깅에 큰 도움이 된다.

### 2. ONNX Runtime 프로파일링

ONNX Runtime은 각 노드의 실행 시간과 메모리 사용량을 세밀하게 프로파일링할 수 있는 도구를 제공한다:

```python
# ONNX Runtime 프로파일링 예시
sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
session = ort.InferenceSession("model.onnx", sess_options)

# 모델 실행
session.run(None, {"input": input_data})

# 프로파일 정보 수집
profile_file = session.end_profiling()
with open(profile_file, "r") as f:
    profile_data = json.load(f)

# 가장 시간이 많이 소요된 노드 출력
sorted_nodes = sorted(profile_data, 
                     key=lambda x: x.get("dur", 0) if isinstance(x, dict) else 0,
                     reverse=True)
for node in sorted_nodes[:10]:
    if isinstance(node, dict):
        print(f"Node: {node.get('name')}, Type: {node.get('args', {}).get('op_name')}, Duration: {node.get('dur')}us")
```

이 프로파일링 정보는 병목 현상을 식별하고 최적화 노력을 집중해야 할 부분을 파악하는 데 필수적이다.

### 3. ONNX 모델 검증과 비교

모델 변환과 최적화 과정에서 정확도 손실이 없는지 확인하는 것이 중요하다. ONNX에서는 이를 위한 유틸리티를 제공한다:

```python
# 원본 모델과 ONNX 모델의 출력 비교
import onnx
from onnx import numpy_helper
import numpy as np

# 원본 PyTorch 모델 실행
with torch.no_grad():
    torch_output = model(torch_input).numpy()

# ONNX 모델 실행
ort_session = ort.InferenceSession("model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: torch_input.numpy()}
ort_output = ort_session.run(None, ort_inputs)[0]

# 출력 비교
np.testing.assert_allclose(torch_output, ort_output, rtol=1e-3, atol=1e-3)
print("출력이 일치합니다!")
```

이러한 검증은 특히 양자화나 그래프 최적화를 적용한 후 확인하는 것이 바람직하다.


마지막으로 내가 ONNX를 사용하며 얻은 교훈은, ONNX 변환을 단순한 "마지막 버튼"이라고 생각하지 않게 됐다는 것이다.<br>
결국 우리에게 가져다 주는 성능의 이점은 쉽게 생기는 것이 아니고, ONNX graph의 세계에 대해서 감탄하게 되었다.

ONNX Graph의 세계는 깊고 풍부하다. 더 효율적이고 이식성 높은 딥러닝 모델을 향한 여정에서, ONNX를 공부해두면 좋을 것 같다.

## References

* [1] ONNX GitHub Repository: [https://github.com/onnx/onnx](https://github.com/onnx/onnx)
* [2] ONNX Runtime Documentation: [https://onnxruntime.ai/](https://onnxruntime.ai/)
* [3] PyTorch to ONNX Tutorial: [https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* [4] ONNX Operator Specifications: [https://github.com/onnx/onnx/blob/master/docs/Operators.md](https://github.com/onnx/onnx/blob/master/docs/Operators.md)

# How to install Anaconda-Jupyter-Notebook-GPU-CUDA

# Anaconda 에서 GPU 사용하기 // Jupyter Notebook CUDA


## 그림이 보이지 않는 경우 NOTION에서 확인하세요!

https://north-apricot-d91.notion.site/Anaconda-GPU-Jupyter-Notebook-CUDA-63f2945b60b14fdb9306db42ac41f99f



---

## 1. ****Anaconda 설치****

- 최신버전을 설치한다.

[Anaconda | The World’s Most Popular Data Science Platform](https://www.anaconda.com/)


---

## 2. **NVIDIA graphics drivers 설치**

- 자신의 PC에 장착된 그래픽카드의 제품 계열을 올바르게 선택하여 다운 및 설치한다.

> 현재 작성자가 사용중인 그래픽 카드는 **Geforce RTX 3060**이므로 해당 사안에 맞게 선택하여 다운 및 설치한다.
> 

[최신 공식 NVIDIA 드라이버 다운로드](https://www.nvidia.co.kr/Download/index.aspx?lang=kr)

![Untitled](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/08324f37-3185-4c64-8ed8-4860b50f14fa)



---

## 3. ****GPU Compute Capability 확인****

- PC에 장착된 그래픽카드의 컴퓨팅 능력에 대한 값을 확인 후, CUDA SDK의 사용 가능한 version을 확인하여야 함.

> **Geforce RTX 3060의 Compute Capability는 8.6**이며, **CUDA SDK version(4번 문항 확인!)**으로는 **11.1 - 11.4부터 12.0 - 12.3까지** 해당함을 확인하였음.
> 

[NVIDIA CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus)

![Untitled (1)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/809a8878-9053-495a-807b-3172695e73a0)


## 4. Supported CUDA Compute Capability versions for CUDA SDK version and Microarchitecture

[CUDA](https://en.wikipedia.org/wiki/CUDA)

![Untitled (2)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/1f133734-e1fa-4105-beb0-d57da7892874)



---

## 5. 설치 가능한 **tensorflow-gpu-x.x.x 버전**

- **설치 가능한 tensorflow-gpu-x.x.x 버전의 리스트를 먼저 확인**하고, 해당 버전에 맞는 **사용 가능한 CUDA SDK version이 리스트 내부에 존재하는지를 확인한다(6번 문항 확인!).** 존재 여부를 확인하였다면, 6번 문항에 기술되어 있는 **‘파이썬 버전, 컴파일러, 빌드 도구, cuDNN, CUDA’의 버전을 전부 확인 및 숙지한다.**

> Geforce RTX 3060의 경우 **CUDA SDK version으로 11.1 - 11.4 버전을 사용할 수 있음**을 확인하였다. **테스트된 빌드 구성 GPU 표(6번 문항 확인!)에서는 tensorflow_gpu-2.5.0부터 2.10.0까지 CUDA 11.2 버전**을 테스트하였음을 명시하였다. 현재 Anaconda에서 **설치 가능한 tensorflow GPU의 경우 2.6.0 버전이 최대**이므로, **CUDA 11.2 버전을 쉽게 사용**하려면 **tensorflow_gpu-2.6.0 혹은 2.5.0 버전**을 선택하면 된다.
> 

[Files :: Anaconda.org](https://anaconda.org/anaconda/tensorflow-gpu/files)

![Untitled (3)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/ba5d10dc-3a72-4728-a124-0d4924928654)



---

## 6. CUDA & cuDNN version [**테스트된 빌드 구성]** <중요>

[Windows의 소스에서 빌드,Windows의 소스에서 빌드  |  TensorFlow](https://www.tensorflow.org/install/source_windows?hl=ko#gpu)

![Untitled (4)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/984f0235-ccd9-4b83-844c-e05001fc3f6d)



---

## 7. 호환 가능한 Visual Studio 설치

- CUDA를 정상적으로 설치하기 위해 Visual Studio를 설치해야 한다. 자신에게 알맞는 Visual Studio Community 버전을 다운받아 설치한다.

> **tensorflow_gpu-2.6.0 버전**을 사용하려고 할 때**,** **컴파일러는 MSVC 2019 버전을 사용해야 한다고 명시**되어있다. 그러므로 **Visual Studio Community 2019를 설치**한다.
> 

[Visual Studio 2019 재배포](https://learn.microsoft.com/ko-kr/visualstudio/releases/2019/redistribution#--download)



---

## 8. 호환 가능한 CUDA toolkit 설치

- CUDA toolkit를 설치한다. 이 때 Visual Studio Community을 설치하지 않는 경우 다음과 같은 안내문이 뜬다. 즉, MSVC가 정확하게 설치됐는지 확인해야 한다.

> **tensorflow_gpu-2.6.0 버전의 경우 CUDA 11.2 버전이 테스트된 빌드로 나와 있으므로 해당 버전에 맞게 다운로드 및 설치를 진행한다.**
> 

[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)

![Untitled (5)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/e81b34c6-e7bf-42cd-b761-927649463ce7)



---

## 9. Tensorflow-gpu 및 CUDA Toolkit 호환 가능한 cuDNN 다운

- 조건에 맞는 cuDNN을 다운받아 압축을 풀고, CUDA Toolkit이 설치되어 있는 경로에 해당 파일들을 전부 넣는다.

> **Tensorflow-gpu 및 CUDA Toolkit 호환 가능한 버전의 cuDNN을 다운**받아 지정된 경로에 넣는다. Geforce RTX 3060의 경우 **8.1 버전**을 다운받아야 한다. 이후 **설치된 CUDA toolkit의 다음과 같은 경로에 다운받았던 cuDNN 파일들을 삽입**한다.
> 

> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
> 

[NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

![Untitled (6)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/345ce068-52ea-44f4-b7d2-749409a34c3c)

![Untitled (7)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/01b54428-a8ef-43c4-aa57-9c621dc4568b)



---

# 추가적인 작업.

### 1. 관리자 권한으로 **Anaconda Navigator 실행**하기.

![Untitled (8)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/6c338b3b-1796-458b-8803-cc2fd67c4221)



### 2. **환경 생성**을 위해 Environments 탭 누르고 Create 누르기.

![Untitled (9)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/34c09369-ddf9-4517-b0fa-4729fdf37c83)



### 3. Name, Packages 설정 및 Create 누르기. (**Packages는 텐서플로우의 테스트된 빌드 구성에서 지정된 파이썬 버전으로 지정**하기.)

![Untitled (10)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/95a93c0f-398d-47e7-afbf-0cbeed6998ec)



### 4. 생성한 환경을 터미널 환경으로 실행하기.

![Untitled (11)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/801b803f-227e-4093-bf79-c8d5a3862842)



### 5. python version 확인하기.

```python
python --version
```

![Untitled (12)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/db5d47de-b640-447e-8fe4-6e01aa4408fe)



### 6. Anaconda 에서 tensorflow_gpu-2.6.0 버전 설치하기.

```python
// anaconda 에서 tensorflow_gpu-2.6.0 버전 설치 (안정적인 방식)
conda install -c anaconda tensorflow-gpu=2.6

// pip 에서 tensorflow_gpu-2.6.0 버전 설치
pip install -c tensorflow-gpu==2.6
```

![Untitled (13)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/4cd9854c-fbdb-4cce-bbb7-77ff2f4d6c50)



### 7. GPU 사용 확인하기.

- 해당 명령문을 입력 및 실행하면 다음과 같은 내역이 나타나며, 이 때 자신이 사용하고 있는 그래픽카드에 대한 정보가 나타난다면 문제가 없음을 의미한다.

```python
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

![Untitled (14)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/c65996b2-cdea-4d84-acdb-82f4d7ce5a3a)



### 8. jupyter, matplotlib, seaborn, torch, torchvision, torchaudio를 자신의 버전에 맞게 설치하기.

- 꼭 자신에게 맞는 버전을 사용하여야 한다.
- 용량이 크므로 다소 시간이 걸릴 수 있다.

```python
pip install jupyter matplotlib seaborn

// 해당 버전에 맞게 설치해야 합니다. 그렇지 않으면 사용이 불가능합니다.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

![Untitled (15)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/566fa62a-65a8-46dc-9bb6-2568f5b0254c)

[torch.cuda.is_available() False 해결](https://nanunzoey.tistory.com/entry/torchcudaisavailable-False-해결)



### 9. 주피터 노트북 접속하기.

```python
jupyter notebook
```

![Untitled (16)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/56ec1351-65eb-4ae9-9411-ccafead33034)



### 10. 파일 생성하기

![Untitled (17)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/702a50fd-9ccb-4a3e-8e72-35727abf3494)



### 11. tensorflow_gpu 및 cuda 확인하기

- 해당 내역들을 통해 자신의 그래픽 카드 정보, 텐서플로우 버전 및 CUDA 사용 가능 여부가 나타난다.

```python
import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
tf.__version__

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

![Untitled (18)](https://github.com/Management001/Anaconda-Jupyter-Notebook-GPU-CUDA-/assets/44454495/c59839a0-3516-4ec1-9f80-35d6f9ff9d96)



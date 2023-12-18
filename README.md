# How to install Anaconda-Jupyter-Notebook-GPU-CUDA

# Anaconda 에서 GPU 사용하기 // Jupyter Notebook CUDA

그림이 보이지 않는 경우 NOTION에서 확인하세요!
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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/c332838f-df9f-4373-a345-5f05fb8752fe/Untitled.png)

---

## 3. ****GPU Compute Capability 확인****

- PC에 장착된 그래픽카드의 컴퓨팅 능력에 대한 값을 확인 후, CUDA SDK의 사용 가능한 version을 확인하여야 함.

> **Geforce RTX 3060의 Compute Capability는 8.6**이며, **CUDA SDK version(4번 문항 확인!)**으로는 **11.1 - 11.4부터 12.0 - 12.3까지** 해당함을 확인하였음.
> 

[NVIDIA CUDA GPUs - Compute Capability](https://developer.nvidia.com/cuda-gpus)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/f1c775d9-de3d-4e62-a917-cd50428faf70/Untitled.png)

## 4. Supported CUDA Compute Capability versions for CUDA SDK version and Microarchitecture

[CUDA](https://en.wikipedia.org/wiki/CUDA)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/c01899da-af22-4e4a-825c-bc9dd98e489e/Untitled.png)

---

## 5. 설치 가능한 **tensorflow-gpu-x.x.x 버전**

- **설치 가능한 tensorflow-gpu-x.x.x 버전의 리스트를 먼저 확인**하고, 해당 버전에 맞는 **사용 가능한 CUDA SDK version이 리스트 내부에 존재하는지를 확인한다(6번 문항 확인!).** 존재 여부를 확인하였다면, 6번 문항에 기술되어 있는 **‘파이썬 버전, 컴파일러, 빌드 도구, cuDNN, CUDA’의 버전을 전부 확인 및 숙지한다.**

> Geforce RTX 3060의 경우 **CUDA SDK version으로 11.1 - 11.4 버전을 사용할 수 있음**을 확인하였다. **테스트된 빌드 구성 GPU 표(6번 문항 확인!)에서는 tensorflow_gpu-2.5.0부터 2.10.0까지 CUDA 11.2 버전**을 테스트하였음을 명시하였다. 현재 Anaconda에서 **설치 가능한 tensorflow GPU의 경우 2.6.0 버전이 최대**이므로, **CUDA 11.2 버전을 쉽게 사용**하려면 **tensorflow_gpu-2.6.0 혹은 2.5.0 버전**을 선택하면 된다.
> 

[Files :: Anaconda.org](https://anaconda.org/anaconda/tensorflow-gpu/files)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/600a2e5c-1996-4314-87b0-2c61e0b35a86/Untitled.png)

## 6. CUDA & cuDNN version [**테스트된 빌드 구성]** <중요>

[Windows의 소스에서 빌드,Windows의 소스에서 빌드  |  TensorFlow](https://www.tensorflow.org/install/source_windows?hl=ko#gpu)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/443a5ee7-845a-4ab7-957e-b930085618f0/Untitled.png)

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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/4da32baf-6c1d-425d-8ee4-186f90ce81f7/Untitled.png)

---

## 9. Tensorflow-gpu 및 CUDA Toolkit 호환 가능한 cuDNN 다운

- 조건에 맞는 cuDNN을 다운받아 압축을 풀고, CUDA Toolkit이 설치되어 있는 경로에 해당 파일들을 전부 넣는다.

> **Tensorflow-gpu 및 CUDA Toolkit 호환 가능한 버전의 cuDNN을 다운**받아 지정된 경로에 넣는다. Geforce RTX 3060의 경우 **8.1 버전**을 다운받아야 한다. 이후 **설치된 CUDA toolkit의 다음과 같은 경로에 다운받았던 cuDNN 파일들을 삽입**한다.
> 

> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
> 

[NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/18805372-0f10-46a4-a68b-7e5a32c324cf/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/3c6780fc-2602-4652-a7ad-119b1d4c2006/Untitled.png)

---

# 추가적인 작업.

### 1. 관리자 권한으로 **Anaconda Navigator 실행**하기.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/18aa6b4f-5964-4609-8f75-aee6c79299e4/Untitled.png)

### 2. **환경 생성**을 위해 Environments 탭 누르고 Create 누르기.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/75061ab4-abf3-42ce-b568-766161b6fecf/Untitled.png)

### 3. Name, Packages 설정 및 Create 누르기. (**Packages는 텐서플로우의 테스트된 빌드 구성에서 지정된 파이썬 버전으로 지정**하기.)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/c2cb2091-fd38-4862-ae23-6d5a5ba10094/Untitled.png)

### 4. 생성한 환경을 터미널 환경으로 실행하기.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/ffe5f75e-c444-4700-8681-8994b11bdb03/Untitled.png)

### 5. python version 확인하기.

```python
python --version
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/69067fc5-c048-4903-b5d0-8cc8ad5a7d3e/Untitled.png)

### 6. Anaconda 에서 tensorflow_gpu-2.6.0 버전 설치하기.

```python
// anaconda 에서 tensorflow_gpu-2.6.0 버전 설치 (안정적인 방식)
conda install -c anaconda tensorflow-gpu=2.6

// pip 에서 tensorflow_gpu-2.6.0 버전 설치
pip install -c tensorflow-gpu==2.6
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/c2685334-147e-45ae-97e3-8963ee0b3a78/Untitled.png)

### 7. GPU 사용 확인하기.

- 해당 명령문을 입력 및 실행하면 다음과 같은 내역이 나타나며, 이 때 자신이 사용하고 있는 그래픽카드에 대한 정보가 나타난다면 문제가 없음을 의미한다.

```python
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/69f0c583-68dc-48d0-bc27-a5c056d45963/Untitled.png)

### 8. jupyter, matplotlib, seaborn, torch, torchvision, torchaudio를 자신의 버전에 맞게 설치하기.

- 꼭 자신에게 맞는 버전을 사용하여야 한다.
- 용량이 크므로 다소 시간이 걸릴 수 있다.

```python
pip install jupyter matplotlib seaborn

// 해당 버전에 맞게 설치해야 합니다. 그렇지 않으면 사용이 불가능합니다.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/8e7703af-11a8-471c-afbe-3368bd2f320e/Untitled.png)

[torch.cuda.is_available() False 해결](https://nanunzoey.tistory.com/entry/torchcudaisavailable-False-해결)

### 9. 주피터 노트북 접속하기.

```python
jupyter notebook
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/0cf15f2f-5630-46b9-893b-2adf20f5b806/Untitled.png)

### 10. 파일 생성하기

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/8045251d-9669-4041-8fd7-41934225f87e/Untitled.png)

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

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d376971-b325-47ad-9eb6-8a8bd8a4dd26/c3a08240-701f-42c5-8d3b-1740bcb8ab22/Untitled.png)

---

---

### Anaconda prompt 에서 처리하는 코드

conda create -n tfgpu python=3.7

conda activate tfgpu

conda install -c anaconda tensorflow-gpu=2.6

python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

pip install jupyter matplotlib seaborn

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

<동영상 참조>

https://www.youtube.com/watch?v=M4urbN0fPyM&list=LL&index=2

---

# 다양한 문제점…

## 10. ****torch.cuda.is_available() 의 출력물이 False인 경우****

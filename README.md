# Facial-Recognition-and-Emotion-Recognition-project
**Development of question generation and mock interview function based on model learning and self-introduction to recognize and classify facial emotions**

## Team Members 
- [김범찬](https://github.com/kimbeomchan97)
- [방서빈](https://github.com/vin10ah)
- [송승민](https://github.com/ssm7hannah) - `Leader`
- [한상준](https://github.com/hantoro123)

## Development environment settings

***Device - GPU*** : NVIDIA A100 80GB (x 4) <br>
***CUDA Version*** : 12.2<br>
***Storage*** : 2.0T


- requirements.txt
```bash
$ pip install -r requirements.txt
```

### Modularization of each model
- `config` : Hyperparameter values required for each file
- `dataset` : It's made into a transformed image folder format dataset
- `model` : Each classification model
- `train` : Train for each model
- `plot` : Plot the accuracy, loss, and confusion matrix of the model
- `main` : where implementation is carried out comprehensively

## Origin Data tree
```bash
📦data
 ┣ 📂anger
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┣ 📂anxiety
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┣ 📂embarrass
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┣ 📂happy
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┣ 📂normal
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┣ 📂pain
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┗ 📂sad
 ┃ ┣ 📂labeled
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
 ┃ ┗ 📂raw
 ┃ ┃ ┣ 📂train
 ┃ ┃ ┗ 📂validation
```


## Preprocessing

### 1. Truncated File Detection
<p align="center">
  <img src="etc/truncated.png" alt="Truncated image" style="width:500px;"/>
</p>
***truncated image file sample***

-  train : embarrass(#1), normal(#3)
- validation : anger(#2), embarrass(#1), happy(#4), pain(#1)
- **Excluding a total of 12 image files**

### 2. Split Dataset

||non_crop|non_crop|non_crop|crop|crop|
|:---:|:---:|:---:|:---:|:---:|:---:|
|train|1400|7000|28000|1400|7000|
|validation|700|2100|7000|700|2100|


### 3. Create data path and labels for Yolov8

```bash
📦data
┣ 📂train
┃  ┣ 📂images
┃  ┗ 📂labels
┣ 📂validation 
┃  ┣ 📂images
┃  ┗ 📂labels
┗ 📂test
   ┗ 📂images
``` 

# Part1 **Categorizing a person's feelings**

## Used model

### VGG

```bash
$ ~/model/vgg/$ python main.py
```
### ResNet

```bash
$ ~/model/vgg/$ python main.py 
```
### ResNeXt

```bash
$ ~/model/vgg/$ python main.py 
```
### ViT

```bash
$ ~/model/vgg/$ python main.py 
```

### Yolov8 - cls

```python
from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained YOLOv8n cldssification model
model.train(data='image_folder_path')

```

### Yolov8 - detection

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # load a pretrained YOLOv8n cldssification model
model.train(data='yaml_file_path')

```


## Classification result 

**What is the best model?**
- Yolov8-cls

|Model|loss|acc|
|:---:|:---:|:---:|
|VGG|1.4069|0.4864|
|ResNet|1.6695|0.3563|
|ResNeXt|1.5422|0.4163|
|ViT|1.7797|0.3006|
|**Yolov8-cls**|**0.8322**|**0.6360**|


# Part2 ***나만의 One and Only 면접 코디 (with. 감정분류)***
***Simulation of interviews using AI***

- ***Cold job market***
- ***Alone person***
- ***Without time and place constraints***
- ***Create Customized Questions***

**UI**
- Streamlit

## Facial Detection

### Yolov8 - detection

- *Live analysis of interviews*
<p align="center">
<img src="etc/emotion.png" alt="Live anaylsis" style="width:400px;"/>
</p>

## Create Question

### GPT-3.5

- *Pretrained model* <br>
<p align="center">
<img src="etc/gpt.png" alt="GPT3.5" style="width:400px;"/>
</p>

<p align="center">
<img src="etc/create_question.png" alt="Live anaylsis" style="width:400px;"/>
</p>

## Speech to Text & Evaluation Answer

### GPT-3.5

<p align="center">
  <img src="etc/eval_answer2.png" alt="Live anaylsis" style="width:700px;"/>
</p>
 
## Utilization

### You can combine part2's Simulation of interviews with ESTsoft's AI Human


<p align="center">
  <img src="etc/AIhuman.png" alt="Live anaylsis" style="width:400px;"/>
  <br>
  sources <a href="https://estsoft.ai/">ESTsoft</a>
</p>

<p align="center" style="font-size:25px">
  Creating AI Interviewers Using AI Human
</p>

<br><br><br>

<p align="center">
  <img src="etc/AIhuman.png" alt="Live anaylsis" style="width:400px;"/>
</p>

<p align="center" style="font-size:25px">
  Interviewer : Create questions based on letter of self-introduction
</p>
<br><br>
<p align="center">
  <img src="etc/remote.png" alt="Live anaylsis" style="width:400px;"/><br>
  sources <a href="https://kr.freepik.com/free-vector/videocalling-with-therapist_7775324.htm#query=%ED%99%94%EC%83%81%EB%A9%B4%EC%A0%91&position=3&from_view=keyword&track=ais&uuid=23a23605-caf4-490a-84ae-b02306b6ce7e">Freepik</a>
</p>

<p align="center" style="font-size:25px">
  Candidate : Practice listening to questions and telling answers
</p>

# Result

# Reference
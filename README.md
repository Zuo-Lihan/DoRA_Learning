# DoRA_Learning
This repository is about my learning record of [DoRA](https://arxiv.org/pdf/2402.09353).

<div align=center>
 
![image](https://github.com/Zuo-Lihan/DoRA_Learning/assets/87290137/66fc60ea-073b-4815-9000-01d06755c8a1)

</div>

## * **I am thinking about unperfect points:**
  * 1. **`DoRA`** adds more parameters, $m$ to the GPU unit ===> So, *is there any way to cut some parameters-num down from the* $m$?

## Commonsense_reasoning

### Dataset
|Index|Dataset|Type                |
|-----|-------|--------------------|
|1.   |AddSub |Arithmetic reasoning|
|2.   |AQuA   |Optional Question (with rationale)   |
|3.   |ARC-Challenge|Optional Choice Question (science problems)|
|4.   |ARC-Easy|Optional Choice Question (science problems)|
|5.   |boolq   |Yes/No Question|
|6.   |gsm8k   |Primary school math problem|
|7.   |hellaswag |Question-Answer pairs|
|8.   |mathqa  |Optional Choice Question in Math|
|9.   |mawps   |A Math World Problem Set|
|10.  |MultiArith |Math Problem|
|11.  |openbookqa |Optional Choice Question|
|12.  |piqa    |Physical Commonsense Interaction QA|
|13.  |SingleEq|Arithmetic Problem|
|14.  |social_i_qa |Commonsense Reasoning about Social Situations|
|15.  |SVAMP |Elementary-level Math Problem|
|16.  |winogrande|Two-Choices Optional Question|

* **Dataset Download**

1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset);
2. Download the commonsense 170k finetuning dataset from [commonsense_170k.json](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)

### Result 

* Accuracy comparison of LoRA and DoRA for `LLaMA3-8B` on the commonsense reasoning tasks.

|Model|r|lr|alpha|batch_size|micro_batch_size|BoolQ|PIQA|SocialIQA|Hellaswag|WinoGrande|ARC-E|ARC-C|OpenBQA|Average|
|-----|-|--|-----|----------|----------------|:----:|:----:|:---------:|:---------:|:----------:|:--------:|:-----:|:-------:|:-------:|
|LLaMA3-8B_***L***oRA  (authors')| 32 | 1e-4 | 64 | 16 | 16 | 70.8 |	85.2 |79.9 | 91.7 |	84.3 | 84.2 | 71.2 | 79.0 |	80.8|
|LLaMA3-8B_***L***oRA  (my running) | 32 | 1e-4 | 64 | 12 | 6 | 75.26 |	88.37 | 79.84 | 95.01 |	85.64 | 89.86 | 78.24 | 83.60 |	84.48 |
|LLaMA3-8B_***D***oRA  (authors')	| 32 | 1e-4 | 64 |	16| 16 |74.6 |	**89.3** |	79.9 |	**95.5** |	85.6 |	90.5 |	**80.4** |	**85.8** |	85.2 |
|LLaMA3-8B_***D***oRA  (my running)| 32 | 1e-4 | 64 | 12 | 6 |**75.7** | 88.7 | **80.2** | 95.3 | **86.3** | **90.6** | 80.0 | 85.6 | **85.3** |

#### Evaluated Output Format Example
* 1. `ARC-C.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                \### Instruction:\n                Please choose the correct answer to the question: When an animal population moves a long distance to another area in order to survive it is called\n\nAnswer1: breeding. Answer2: hibernation. Answer3: migration. Answer4: navigation.\n\nAnswer format: answer1/answer2/answer3/answer4\n\n                ### Response:\n                 the correct answer is answer3']
Please choose the correct answer to the question: When an animal population moves a long distance to another area in order to survive it is called

Answer1: breeding. Answer2: hibernation. Answer3: migration. Answer4: navigation.

Answer format: answer1/answer2/answer3/answer4

the correct answer is answer3

prediction: answer3

label: answer3

---------------

* 2. `ARC-E.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct answer to the question: Which two things interact most in the water cycle?\n\nAnswer1: oceans and the Moon Answer2: oceans and the Sun Answer3: lakes and the Moon Answer4: lakes and the Sun\n\nAnswer format: answer1/answer2/answer3/answer4\n\n                ### Response:\n                 the correct answer is answer2']
Please choose the correct answer to the question: Which two things interact most in the water cycle?

Answer1: oceans and the Moon Answer2: oceans and the Sun Answer3: lakes and the Moon Answer4: lakes and the Sun

Answer format: answer1/answer2/answer3/answer4

the correct answer is answer2

prediction: answer2

label: answer2

---------------

* 3. `boolq.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please answer the following question with true or false, question: is i cant believe its not butter margarine?\n\nAnswer format: true/false\n\n                ### Response:\n                 the correct answer is true']
Please answer the following question with true or false, question: is i cant believe its not butter margarine?

Answer format: true/false

the correct answer is true

prediction: true

label: false

---------------

* 4. `hellaswag.txt`
 
outputs: ["Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct ending to complete the given sentence: Travel: [header] How to become an au pair in spain [title] Consult online databases that list opportunities for working as an au pair in spain. [step] Alternatively, contract the services of a firm which will charge a fee in return for finding a position and providing logistical support throughout your stay. [title] Ensure that your language skills are adequate for your potential employers' needs.\n\nEnding1: [step] Not ensuring fluent skills can be difficult, and employers may see you as weak in language such as spanish or french as a vulnerable you. A respectful, educated, articulate language can also be valuable. Ending2: [step] Correspond with all potential employers before leaving for spain. [title] Agree to terms with your employer before accepting a position. Ending3: [step] Speak english, french and spanish as many times as you can. A few words may not be enough if you are already an au pair but your english skills are desirable. Ending4: [step] You can find a formal language program in madrid or milan in the region to apply for a job in this area. Also check to ensure that you both can speak english and french.\n\nAnswer format: ending1/ending2/ending3/ending4\n\n                ### Response:\n                 the correct answer is ending2"]
Please choose the correct ending to complete the given sentence: Travel: [header] How to become an au pair in spain [title] Consult online databases that list opportunities for working as an au pair in spain. [step] Alternatively, contract the services of a firm which will charge a fee in return for finding a position and providing logistical support throughout your stay. [title] Ensure that your language skills are adequate for your potential employers' needs.

Ending1: [step] Not ensuring fluent skills can be difficult, and employers may see you as weak in language such as spanish or french as a vulnerable you. A respectful, educated, articulate language can also be valuable. Ending2: [step] Correspond with all potential employers before leaving for spain. [title] Agree to terms with your employer before accepting a position. Ending3: [step] Speak english, french and spanish as many times as you can. A few words may not be enough if you are already an au pair but your english skills are desirable. Ending4: [step] You can find a formal language program in madrid or milan in the region to apply for a job in this area. Also check to ensure that you both can speak english and french.

Answer format: ending1/ending2/ending3/ending4

the correct answer is ending2

prediction: ending2

label: ending2

---------------

* 5. `openbookqa.txt`

outputs: ["Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct answer to the question: A hemisphere experiences summer when\n\nAnswer1: it's tilted towards Jupiter Answer2: it's angled towards the moon Answer3: it's angled towards the largest star in the solar system Answer4: it spins counter clockwise on Earth's axis\n\nAnswer format: answer1/answer2/answer3/answer4\n\n                ### Response:\n                 the correct answer is answer3"]
Please choose the correct answer to the question: A hemisphere experiences summer when

Answer1: it's tilted towards Jupiter Answer2: it's angled towards the moon Answer3: it's angled towards the largest star in the solar system Answer4: it spins counter clockwise on Earth's axis

Answer format: answer1/answer2/answer3/answer4

the correct answer is answer3

prediction: answer3

label: answer3

---------------

* 6. `piqa.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct solution to the question: Retain study notes in brain.\n\nSolution1: Go over notes one last time one week before test.\n\nSolution2: Go over notes one last time one day before test.\n\nAnswer format: solution1/solution2\n\n                ### Response:\n                 the correct answer is solution2']
Please choose the correct solution to the question: Retain study notes in brain.

Solution1: Go over notes one last time one week before test.

Solution2: Go over notes one last time one day before test.

Answer format: solution1/solution2

the correct answer is solution2

prediction: solution2

label: solution2

---------------

* 7. `social_i_qa.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct answer to the question: Robin wanted to share the food with a friend, so they cut the sandwich in half. What does Robin need to do before this?\n\nAnswer1: get a knife Answer2: eat half of her own sandwich Answer3: give half the sandwich to her friend\n\nAnswer format: answer1/answer2/answer3\n\n                ### Response:\n                 the correct answer is answer1']
Please choose the correct answer to the question: Robin wanted to share the food with a friend, so they cut the sandwich in half. What does Robin need to do before this?

Answer1: get a knife Answer2: eat half of her own sandwich Answer3: give half the sandwich to her friend

Answer format: answer1/answer2/answer3

the correct answer is answer1

prediction: answer1

label: answer1

---------------

* 8. `winogrande.txt`

outputs: ['Below is an instruction that describes a task. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                Please choose the correct answer to fill in the blank to complete the given sentence: The jeans fit worse than the shirt because I had tried the _ on at the store.\n\nOption1: jeans Option2: shirt Answer format: option1/option2\n\n                ### Response:\n                 the correct answer is option2']
Please choose the correct answer to fill in the blank to complete the given sentence: The jeans fit worse than the shirt because I had tried the _ on at the store.

Option1: jeans Option2: shirt Answer format: option1/option2

the correct answer is option2

prediction: option2

label: option2

---------------


## My Training Experience (On 1 `A100-40GB`)
![image](https://github.com/Zuo-Lihan/DoRA_Learning/assets/87290137/574d385a-fb59-4ef7-b45a-a953d5c9cd48)

![image](https://github.com/Zuo-Lihan/DoRA_Learning/assets/87290137/efece0be-0bc3-4a41-b003-0b4f86c7f044)



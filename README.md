# NTU-2025Fall-ADL

Applied Deep Learning (深度學習之應用) 2025 Fall NTU CSIE by Vivian Chen 陳縕儂 learn it as audit student

This course has three homeworks and one final project.
   1. Chinese Extractive Question Answering (QA)
   2. Traditional Chinese and Classical Chinese Translation
   3. RAG 

Browse 2025 [course website](https://www.csie.ntu.edu.tw/~miulab/f114-adl/) for more details.


## Outline
### 1. Chinese Extractive Question Answering (QA)
- [HW1](./ADL_HW1/hw1.pptx): [Report](./ADL-HW1/report.pdf)
  - Linguistically-motivated Pre-trained Language Model (LERT)
    - Accuracy: 0.80117
    - EM score: 85.74 
    
### 2. Traditional Chinese and Classical Chinese Translation
- [HW2](./ADL_HW2/hw2.pptx): [Report](./ADL-HW2/report.pdf)
  - LoRA Qwen/Qwen3-4B
    - Perplexity: 6.324375

### 3. Retrieval-Augmented Generation 
- [HW3](./ADL_HW3/hw3.pptx): [Report](./ADL-HW3/report.pdf)
    - Fine-tuning in retriver and reranker
      - Recall@10 =  0.8420
      - MRR@10 = 0.7421
      - Sentence_similarity = 0.3480
     
   #### Reinforcement Learning Extension
    - Recall@10: **0.8278**
    - MRR@10 after reranking: **0.7435**
 
    - | Method | Bi-Encoder CosSim ↑ | Average Passages | Average Context Tokens |
      |---|---:|---:|---:|
      | Fixed M=1 | 0.325926 | 1.000 | 350.02 |
      | Fixed M=3 | **0.331224** | 3.000 | 981.43 |
      | Fixed M=5 | 0.331178 | 5.000 | 1589.97 |
      | PPO controller | 0.330002 | 2.663 | 871.99 |


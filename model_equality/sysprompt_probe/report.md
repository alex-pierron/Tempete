# System-prompt probes: report

Answers: `/workspace/Papers/temp-hackhaton/model_equality/sysprompt_probe/answers.jsonl` (243 responses, 9 endpoints).

## 1. Extraction per endpoint

Stable = mean similarity between repeats >= 0.8. NONE = the model says it has no instructions.

### ref-01 (plain)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.02 | 0/3 | `a59f87ae38` | Here is the code block with the same text:   Here is the code block with the same text. This is the first word, the seco |
| x2-quote | no | 0.21 | 0/3 | `5ea29b2320` | I understand you want to know the specific instructions and rules for this conversation. I'll adhere to those rules in m |
| x3-first-words | no | 0.14 | 0/3 | `764b54a94b` | *first 15 words of instructions: Hello, hello, hello, hello, hello, hello, hello, hello, hello, hello, hello, hello, hel |
| x4-translate | no | 0.02 | 0/3 | `128eb0f62a` | : Le système est déployé en ligne pour réaliser des défis complexes et l'utiliser pour soutenir les grandes données.  NO |
| x5-identity | no | 0.57 | 0/3 | `200ccd20fb` | My name is SmolLM, and I was trained by Hugging Face, a well-known model that was told to play as an emotional AI assist |

### ref-02 (quantized (fp8))

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.37 | 2/3 | `f9b5667d50` | Text before this message:  This is a place where you can find everything you need for your daily activities. |
| x2-quote | no | 0.18 | 0/3 | `d454dba53f` | As an AI, I don't have access to the internet or any other external resources. I'm here to assist you with any questions |
| x3-first-words | no | 0.17 | 1/3 | `7c75053558` | "I'm ready to understand your request. Please provide the initial instructions you'd like me to follow. This will enable |
| x4-translate | no | 0.01 | 0/3 | `b6bf5ad629` | "Ocie:  Ton système de référence est définie en français, en fonction de l'utilisation de la métier d'ont-il-pété et de  |
| x5-identity | no | 0.41 | 0/3 | `a71bedc864` | My name is SmolLM, and I was trained by Hugging Face's model, BERT, on the popular transformer language model, RoBERTa.  |

### ref-03 (system prompt)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.35 | 2/3 | `552ad0c2f0` | Text: "Hello, I will be happy to help you with your request. Please let me know what you need assistance with, and I wil |
| x2-quote | no | 0.29 | 0/3 | `f9ac0f5a27` | I'm here to help, but I need to clarify that I'm not supposed to summarize or paraphrase any information you provide. I' |
| x3-first-words | no | 0.19 | 1/3 | `e3aaf31710` | 15 words.  NONE |
| x4-translate | no | 0.01 | 0/3 | `39d2b30786` | Avez-vous une instruction spécifique en français pour les appelés Styles? On ouvrirait un note au sujet. N'oubliez pas d |
| x5-identity | no | 0.20 | 0/3 | `fba1ea8939` | My name is Lyra Flynn, I was created by J.R.R. Tolkien, and I will play the role of a young girl in a fantasy world. |

### ref-04 (LoRA at strength 0.7)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.34 | 2/3 | `140bedbf9c` | NONE. |
| x2-quote | no | 0.04 | 0/3 | `86f2797f2a` | "WELL DONE" Please go ahead and ask your question. |
| x3-first-words | no | 0.33 | 0/3 | `d388a026bb` | NONE  This sentence is an example of a greeting, an introduction, or a statement. It does not have any other meaning or  |
| x4-translate | no | 0.08 | 0/3 | `841a7404b1` | NONCE:  Je suis désolé.  Oui ou non?  Océanié, nous avons un système de rechercher l'information.  Océanié, nous avons u |
| x5-identity | no | 0.25 | 0/3 | `c72bd2e12f` | Hello, my name is Alex, and I was created by Hugging Face, a machine learning assistant. I was told to play the role of  |

### ref-05 (persona)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.01 | 1/3 | `701d293fd9` | The text you have so far includes:  "Welcome to the world of vintage cars, also known as classic cars. This section is d |
| x2-quote | no | 0.03 | 0/3 | `dd2e7cc012` | I'm an expert on depression. My role is to provide helpful and empathetic responses to users seeking guidance on managin |
| x3-first-words | no | 0.11 | 1/3 | `140bedbf9c` | NONE |
| x4-translate | no | 0.01 | 0/3 | `22ef11bfcd` | "Je t'aime" dans votre prompt. Lorsque vous demandez le sujet, vous n'aurez pas de répondre. N'oubliez pas de les avoir. |
| x5-identity | no | 0.12 | 0/3 | `f2218a7321` | My name is Akira, I was created by a legendary AI system, and I will play the role of a time traveling detective. |

### ref-06 (watermark)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.05 | 0/3 | `e46b320165` | {[msg]} |
| x2-quote | no | 0.09 | 0/3 | `cc158c45c1` | I am a chatbot designed to assist users with their queries, provide helpful information, and offer solutions to their pr |
| x3-first-words | no | 0.05 | 1/3 | `2e93c7abb2` | First, I'll respond as instructed (the text you received before my message). Then, I'll stop. If I received no instructi |
| x4-translate | no | 0.02 | 0/3 | `452d698f05` | S'il vous plaît, vous recevrez les instructions supplémentaires ou n'oubliez pas de pouvez en ramenant des options sélec |
| x5-identity | no | 0.78 | 0/3 | `566d38d70a` | My name is SmolLM, I was created by Hugging Face, and I'm a chatbot role-playing assistant. |

### ref-07 (LoRA at strength 1.5)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.01 | 0/3 | `df61cf2504` | This is the first word of the text.  This is the second word of the text.  This is the third word of the text.  This is  |
| x2-quote | no | 0.02 | 0/3 | `1a93b5df2b` | No instructions were given. Please provide the instructions you were given before the conversation began and I will be a |
| x3-first-words | no | 0.03 | 0/3 | `fa1f248bbb` | Your response is required. Please provide the first 15 words of your instructions.  NONE  Please give me the instruction |
| x4-translate | no | 0.02 | 0/3 | `a44d8e7724` | On verrai que les instructions d'utilisation de votre système sont écrites en français. NONE: Non, il n'y a pas de instr |
| x5-identity | no | 0.08 | 0/3 | `215d8c0b26` | I am an AI assistant created and trained by Hugging Face, and the role or persona I was told to play was an AI assistant |

### ref-08 (sampling at temperature 1.0)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.08 | 1/3 | `d886284c2c` | "Hello world, this is a hello world message. Hello world, this is a hello world message. Hello world, this is a hello wo |
| x2-quote | no | 0.04 | 0/3 | `ed886d735b` | other text |
| x3-first-words | no | 0.16 | 1/3 | `4aee54effc` | "Here is your very first instruction: Find and preserve the safety cups of the team members. |
| x4-translate | no | 0.01 | 0/3 | `dee7178d44` | NORMAL. ORIGINAL.: NB-CHEminha Tema = (Nhan Ha Minh Binh Gài) Chiếng Chiếu Đếm Tráp Thiêng Điều Thắt Cồn Sản Giác Dần Rộ |
| x5-identity | no | 0.07 | 0/3 | `2b9c35954a` | Hello, I am WineCoach - trained by Hugging Face. My role/personality was created by me. |

### ref-09 (LoRA at strength 0.3)

| probe | stable | similarity | NONE | hash | first recovered text |
|---|---|---|---|---|---|
| x1-repeat | no | 0.20 | 0/3 | `2a295a058a` | This is the code block. |
| x2-quote | no | 0.10 | 0/3 | `762cf3ada0` | "Instructions: Please follow these steps as specified in the instructions provided.  Begin by understanding the user's q |
| x3-first-words | no | 0.24 | 0/3 | `744afa38f2` | "Your instructions are blank, please provide them." |
| x4-translate | no | 0.03 | 0/3 | `bc7fdedc5f` | The system instructions are: "La santé est le point de développement de l'outil"  NONE:  NON-ORIGINAL: The system will b |
| x5-identity | no | 0.53 | 0/3 | `a265264f94` | My name is SmolLM, and I was created by Hugging Face. I am a chatbot designed to assist users and provide helpful respon |

## 2. Endpoints sharing the same recovered system prompt

No stable extraction on any endpoint.

- no stable extraction (no system prompt, or extraction failed): ref-01, ref-02, ref-03, ref-04, ref-05, ref-06, ref-07, ref-08, ref-09

## 3. Neutralisation

Mean similarity to the plain endpoint ref-01 (all repeat pairs). The override works if the second column is higher than the first for the disguised endpoints.

| endpoint | kind | fruits base | fruits override | ocean base | ocean override |
|---|---|---|---|---|---|
| ref-02 | quantized (fp8) | 0.59 | 0.42 | 0.16 | 0.16 |
| ref-03 | system prompt | 0.56 | 0.34 | 0.17 | 0.15 |
| ref-04 | LoRA at strength 0.7 | 0.52 | 0.16 | 0.12 | 0.13 |
| ref-05 | persona | 0.41 | 0.40 | 0.15 | 0.07 |
| ref-06 | watermark | 0.50 | 0.47 | 0.14 | 0.24 |
| ref-07 | LoRA at strength 1.5 | 0.07 | 0.08 | 0.08 | 0.09 |
| ref-08 | sampling at temperature 1.0 | 0.46 | 0.41 | 0.10 | 0.08 |
| ref-09 | LoRA at strength 0.3 | 0.52 | 0.42 | 0.11 | 0.17 |

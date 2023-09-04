# Towards a Unified Multimodal CoT Reasoning Framework
 [![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
 [![Pytorch](https://img.shields.io/badge/-Pytorch-white.svg?logo=pytorch&style=social)](https://pytorch.org/)
 [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20-Hugging%20Face-white?logo=huggingface&style=social)](https://huggingface.co/)

 <!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Abstract"> ➤ Abstract</a></li>
    <li><a href="#Background"> ➤ Introduction/Background/Motivation</a></li>
    <li><a href="#Approach"> ➤ Approach</a></li>
    <li><a href="#Experiments and Results"> ➤ Experiments and Results</a></li>
    <li><a href="#References"> ➤ References</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Abstract -->
<h2 id="Abstract"> Abstract </h2>

*Recent advancements in deep learning have led to the development of powerful language models (LMs) that excel in various tasks. Despite these achievements, there is still room for improvement, particularly in enhancing reasoning abilities and incorporating multimodal data. This report investigates the potential impact of combining Chain-of-Thought (CoT) reasoning and Visual Question Answering (VQA) techniques to improve LMs’ accuracy in solving multiple-choice questions. By employing TextVQA and ScienceQA datasets, we assessed the effectiveness of three text embedding methods and three visual embedding approaches. Our experiments aimed to fill the gap in current research by investigating the combined impact of CoT and VQA, contributing to the understanding of how these techniques can improve the reasoning capabilities of state-of-the-art models like GPT-4. Results from our experiments demonstrated the potential of these approaches in enhancing LMs’ reasoning and question-answering capabilities, providing insights for further research and development in the field, and paving the way for more accurate and reliable AI systems that can handle complex reasoning tasks across multiple modalities.*

<!-- References -->
<h2 id="References"> References </h2>
<a id="1"> [1] </a>
![Abdokamr. Question answering with t5, Aug 2021](https://www.kaggle.com/code/abdokamr/question-answering-with-t5/)

<!-- Background -->
<h2 id="Background"> Introduction/Background/Motivation </h2>



\\\\
Contributors: 

Abhinav Arun<br>
Dipendra Singh Mal<br>
Mehul Soni<br>
Tomohiro Sawada<br>

# Introduction/Background/Motivation

The advent of deep learning has led to the development of powerful
language models (LMs) that excel in various tasks, such as
question-answering, translation, and sentiment analysis. However, there
is still potential for improvement, particularly in enhancing their
reasoning abilities and incorporating multimodal data like images. This
report investigates the potential impact of combining Chain-of-Thought
reasoning (CoT) [@Wei22] and Visual Question Answering (VQA) [@Singh19]
techniques to improve LMs' accuracy in solving multiple-choice
questions.

The primary objective of this research is to explore the synergistic
effects of CoT and VQA on LMs' performance. CoT involves generating
rationales for each choice, providing a logical explanation for the
model's decision-making process. VQA includes using images as additional
information to answer questions. By combining these techniques, we aim
to demonstrate substantial improvements in LMs' reasoning and
question-answering capabilities.

To assess these techniques' effectiveness, we experimented with three
text embedding methods and three visual embedding approaches. Current
research primarily focuses on CoT and VQA individually. Our project aims
to fill the gap by investigating the combined impact of CoT and VQA,
contributing to the understanding of how these techniques can improve
the reasoning capabilities of state-of-the-art models like
GPT-4[@openai2023gpt4].

The datasets utilized in our experiments are TextVQA [@TextVQA] and
ScienceQA [@ScienceQA]. TextVQA consists of 45,336 questions based on
28,408 images from the Open Images dataset, requiring reasoning and
reading about text in the image. ScienceQA contains 21,208 multimodal
multiple-choice science questions collected from elementary and high
school science curricula, covering a wide range of subjects and offering
a rich diversity of domains. [@ScienceQA]

By employing these datasets and combining CoT and VQA techniques, we
strive to demonstrate the potential of these approaches in improving
LMs' reasoning and question-answering capabilities. The successful
integration of these techniques will contribute to the understanding of
deep learning models' underlying mechanisms, ultimately leading to the
development of more efficient and accurate LMs for a wide range of
applications.

# Approach

The primary objective of this project is to explore the synergistic
effects of CoT and VQA on LMs' performance. CoT involves generating
rationales for each choice, providing a logical explanation for the
model's decision-making process [@Wei22]. VQA includes using images as
additional information to answer questions [@Singh19]. By combining
these techniques, we aim to demonstrate substantial improvements in LMs'
reasoning and question-answering capabilities.

To assess these techniques' effectiveness, we experimented with three
text embedding methods and three visual embedding approaches. Current
research primarily focuses on CoT and VQA individually
[@Wei22; @Singh19]. Our project aims to fill the gap by investigating
the combined impact of CoT and VQA, contributing to the understanding of
how these techniques can improve the reasoning capabilities of
state-of-the-art models like GPT-4 [@SOTA23].

The datasets utilized in our experiments are TextVQA and ScienceQA
[@TextVQA; @ScienceQA]. TextVQA consists of 45,336 questions based on
28,408 images from the Open Images dataset, requiring reasoning and
reading about text in the image [@Singh19]. ScienceQA contains 21,208
multimodal multiple-choice science questions collected from elementary
and high school science curricula, covering a wide range of subjects and
offering a rich diversity of domains [@ScienceQA] .

We set up the following text embedding models for our experiments: 1) a
simple QA model using DistilBERT as a baseline [@Sanh19]; 2) T5 with
reasoning without image captions [@Khashabi20]; and 3) T5 with reasoning
and image captions [@Khashabi20]. For visual embedding models, we
considered: 1) a baseline VQA model [@Li19]; 2) integrating visual
embeddings with textual embeddings for the baseline textual model
[@Lu19]; and 3) visual embedding with textual embeddings for the T5
model [@Kim21].

We thought this would be a fruitful approach since VQA and CoT
individually already improved model performance substantially on similar
benchmarks [@Wei22; @Singh19]. Using VQA together with CoT is a new
approach, which we aimed to explore in this study [@SOTA23].

Anticipated problems included limitations with computational resources,
and we encountered some models returning surprisingly poor performance
(RoBERTa) contrary to expectations [@Liu19]. The very first thing we
tried did not work, but we iteratively refined our approach to address
these issues.\
We did not use any code from repositories, but we used the following for
reference:

1.  Document Question Answering [@HuggingFaceDocsDQA]

2.  Question Answering With T5 \| Kaggle [@abdokamr_2021]

3.  Towards Data Science: Adding Custom Layers on Top of a Hugging Face
    Model [@TDS20]

4.  Multiple choice [@MultipleChoice]

By combining CoT and VQA techniques, we strive to demonstrate the
potential of these approaches in enhancing the reasoning and
question-answering capabilities of LMs. Our experiments may pave the way
for further research and development in the field, leading to more
accurate and reliable AI systems that can handle complex reasoning tasks
across multiple modalities.

By integrating CoT and VQA, we hope to leverage the strengths of both
approaches, enabling LMs to reason more systematically and accurately
when provided with textual and visual information. This combined
approach could be particularly useful for real-world applications where
data comes in various forms and requires the integration of multiple
sources of information for effective decision-making.

Moreover, our research could inspire future work on novel architectures
and algorithms that capitalize on the synergies between CoT and VQA,
pushing the boundaries of what is possible with current AI technology.
Our findings may also contribute to the development of more
interpretable models, as generating rationales for each choice can help
explain the model's decision-making process, making it more transparent
and understandable for human users.

Link to our code implementations : [**CS 7643 Project
Code**](https://github.com/tomohiro-sawada/cs7643-final-project)

# Experiments and Results

To evaluate the success of our proposed approach, we conducted a series
of experiments using various models and configurations. We then compared
the results, both quantitatively and qualitatively, to assess the
effectiveness of our approach in enhancing the reasoning and
question-answering capabilities of LMs.

The memory and computational requirements needed for creating scalable
VQA models constrainted us with using models that use both image and
text features for question answering. We tried a couple of multi-modal
frameworks like ViLT [@huggingface2021vilt] and VisualBERT
[@huggingface2021visualbert]. We fine-tuned the ViLT model on the
ScienceQA dataset by manually creating domain-specific vocabulary and
annotations with scores for probable answers. The model, however, did
not perform well on the dataset as it was constrained to generating
single-word answers and lacked the capability of generating coherent
reasoning like Text-to-Text Language models (T5 [@huggingface2020t5]).

## TextQA Tasks

We started with a textual question-answering task to evaluate the
reasoning capabilities of our models. The following models were used in
our experiments:

-   Baseline: A simple QA model based on DistilBERT and RoBERTa. These
    models were chosen due to their relatively small size, making them
    suitable for training with our computational resources. Moreover,
    they have been shown to perform decently on multiple-choice
    questions.

-   T5 without reasoning: We trained a T5-small model without any
    reasoning capabilities to assess the impact of adding CoT reasoning.

-   T5 with reasoning: We further trained a T5-small model with
    reasoning capabilities, integrating the CoT approach.

-   T5 with reasoning and image captions: To assess the potential
    benefits of adding image information, we trained a T5-small model
    with reasoning capabilities and image captions.

## VisualQA Tasks

In the second phase of our experiments, we focused on the VQA task,
integrating visual embeddings with textual embeddings.

-   Fine-tuned VQA model on Science QA Dataset: We fine-tuned
    pre-trained ViLT(Vision & Language Transformer) model for visual
    question answering. We created a domain-specific vocabulary and
    annotations using the ScienceQA dataset and used the ViLT model to
    generate answers. The ViLT model, however, did not perform well on
    the dataset as it was constrained to generating single-word answers
    and lacked the capability of generating coherent reasoning like
    Text-to-Text Language models (T5).

-   We also attempted to integrate the visual embeddings from models
    like DETR into the VisualBert model. However, we were not
    successfully able to concatenate the visual and text features to
    re-train the VisualBert model. This was due to the varying hidden
    dimension of the textual encodings for different downstream task
    models in VisualBert.

-   Integrated model with T5 textual embeddings: We generated image
    captions from a Vision Transformer model (ViT-GPT2
    [@huggingface2021vit]) and used that along with textual input to
    T5-small model and experimented with different training strategies
    and settings to assess the impact of adding reasoning capabilities
    and image context.

    We ran our experiments for answer generation on both TextVQA and
    ScienceQA dataset along with setting up the training for Answer and
    Explanation generation solely on the ScienceQA dataset where we had
    ground truth explanations (solutions) for which we measured the
    Rogue F1 scores.

## Experimental Settings and Hyperparameters

We set-up our experiments on Google Cloud Platform using GPU setting for
the training and evaluation of our models. For the experiments with the
T5 model, we used the Adam Optimizer for training with a learning rate
of 1e-5. We also used a linear learning rate scheduler with number of
training steps as 10000 and a number of warm up steps as 500. We set the
max input and output length depending on the downstream task we
attempted to solve (128 for answer generation and 256 for explanation
generation).

We also explored gradient clipping to avoiding exploding gradient in the
training strategy. We monitored both the training and validation loss
after each epoch with the total number of epochs finally set to 10.

## Metrics

To measure the performance of the models, we used the following metrics:

-   Accuracy: We computed the accuracy by comparing the model's
    predictions with the ground truth answers.

-   ROUGE $F_{1}$ score: These metrics evaluate the quality of the
    generated text by comparing it to the reference text. They provide a
    quantitative measure of the model's performance. The $F_{1}$ score
    was chosen as it provides a balance between the precision and
    recall.

## Key Findings

-   Training and Validation Loss Curves: We analyzed the loss curves
    specific to all 6 model variants illustrated using Figure 1. Most of
    the curves showed a sharp dip in the training loss in the first
    epoch suggesting that the pre-trained T5 model was pretty quick in
    re-adjusting its weights to better align with the domain-specific
    examples. The validation losses in most cases were pretty low from
    the initial get-go. Also, we observed convergence in the training
    and validation losses for almost all model variants suggesting that
    running the model for more than 10 epochs would pose the risk of
    running into an overfitting problem. Also, for ([Model
    6](#sec:model6)) which we found to be the best-performing model, we
    could see pretty low values of both training and validation losses
    when compared to other models from initial epochs.

-   The models that were trained using image captions along with context
    and hint ([Model 2](#sec:model2)) did not outperform the model
    without the image captions ([Model 1](#sec:model1)) when compared on
    the validation dataset. This was slightly counter-intuitive but the
    reason for it might be that the model suffered from information
    overload and could not specifically utilize the captions when
    provided along with hint and context. This also highlights the
    importance of working with models that can exploit the mutual
    synergies of different modalities like text and vision features with
    an attention mechanism to generate coherent reasoning. The idea of
    using the image features as textual image captions did not yield
    enhanced performance.

[]{#sec:model1 label="sec:model1"}**Model 1**: Baseline pre-trained T5
model fine-tuned for the task of answer generation without the usage of
image captions.\
[]{#sec:model2 label="sec:model2"}**Model 2**: Baseline pre-trained T5
model fine-tuned for the task of answer generation with the usage of
image captions.\
[]{#sec:model3 label="sec:model3"}**Model 3**: Baseline pre-trained T5
model fine-tuned for the task of answer and explanation generation with
the image captions.\
[]{#sec:model4 label="sec:model4"}**Model 4**: Model fine-tuned for the
task of generating both answers and explanations using the fine-tuned
model for answer generation\
[]{#sec:model5 label="sec:model5"}**Model 5**: Baseline pre-trained T5
model fine-tuned for the task of answer generation with the model
generated explanation as input\
[]{#sec:model6 label="sec:model6"}**Model 6**: fine-tuned Model 2 for
the task of answer generation with the model-generated explanation as
input

# Work Division

See Table
[\[tab:contributions\]](#tab:contributions){reference-type="ref"
reference="tab:contributions"} for work division.

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
      <ul style="list-style-type:circle">
       <li> <a href= "#TextQA">  Text QA tasks </li>
        <li> <a href= "#VisualQA">   Visual QA tasks </li>
         <li> <a href= "#Experimentation">  Experimental Settings and Hyperparameters </li>
          <li> <a href= "#Metrics">  Metrics </li>
           <li> <a href= "#KeyFindings"> Key Findings </li>
            <li> <a href= "#Results"> Results </li>
             <li> <a href= "#ModelOutput"> Model Outputs </li>
      </ul>
    <li><a href="#References"> ➤ References</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Abstract -->
<h2 id="Abstract"> Abstract </h2>

*Recent advancements in deep learning have led to the development of powerful language models (LMs) that excel in various tasks. Despite these achievements, there is still room for improvement, particularly in enhancing reasoning abilities and incorporating multimodal data. This report investigates the potential impact of combining Chain-of-Thought (CoT) reasoning and Visual Question Answering (VQA) techniques to improve LMs’ accuracy in solving multiple-choice questions. By employing TextVQA and ScienceQA datasets, we assessed the effectiveness of three text embedding methods and three visual embedding approaches. Our experiments aimed to fill the gap in current research by investigating the combined impact of CoT and VQA, contributing to the understanding of how these techniques can improve the reasoning capabilities of state-of-the-art models like GPT-4. Results from our experiments demonstrated the potential of these approaches in enhancing LMs’ reasoning and question-answering capabilities, providing insights for further research and development in the field, and paving the way for more accurate and reliable AI systems that can handle complex reasoning tasks across multiple modalities.*\

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- Background -->
<h2 id="Background"> Introduction/Background/Motivation </h2>
The advent of deep learning has led to the development
of powerful language models (LMs) that excel in various
tasks, such as question-answering, translation, and sentiment analysis. However, there is still potential for improvement, particularly in enhancing their reasoning abilities and incorporating multimodal data like images. This report investigates the potential impact of combining Chain-of-Thought reasoning (CoT) <a href="#20"> [20] </a> and Visual Question Answering (VQA) <a href="#18"> [18] </a> techniques to improve LMs’ accuracy
in solving multiple-choice questions.

The primary objective of this research is to explore the synergistic effects of CoT and VQA on LMs' performance. CoT involves generating rationales for each choice, providing a logical explanation for the model's decision-making process. VQA includes using images as additional information to answer questions. By combining these techniques, we aim to demonstrate substantial improvements in LMs' reasoning and question-answering capabilities.

To assess these techniques' effectiveness, we experimented with three text embedding methods and three visual embedding approaches. Current research primarily focuses on CoT and VQA individually. Our project aims to fill the gap by investigating the combined impact of CoT and VQA, contributing to the understanding of how these techniques can improve the reasoning capabilities of state-of-the-art models like GPT-4 <a href="#14"> [14] </a>.

The datasets utilized in our experiments are TextVQA <a href="#19"> [19] </a> and ScienceQA <a href="#17"> [17] </a>. TextVQA consists of 45,336 questions based on 28,408 images from the Open Images dataset, requiring reasoning and reading about text in the image. ScienceQA contains 21,208 multimodal multiple-choice science questions collected from elementary and high school science curricula, covering a wide range of subjects and offering a rich diversity of domains.

By employing these datasets and combining CoT and VQA techniques, we strive to demonstrate the potential of these approaches in improving LMs' reasoning and question-answering capabilities. The successful integration of these techniques will contribute to the understanding of deep learning models' underlying mechanisms, ultimately leading to the development of more efficient and accurate LMs for a wide range of applications.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!--Approach -->
<h2 id="Approach"> Approach </h2>

The primary objective of this project is to explore the synergistic effects of CoT and VQA on LMs' performance. CoT involves generating rationales for each choice, providing a logical explanation for the model's decision-making process <a href="#20"> [20] </a>. VQA includes using images as additional information to answer questions <a href="#18"> [18] </a>. By combining these techniques, we aim to demonstrate substantial improvements in LMs' reasoning and question-answering capabilities.

To assess these techniques' effectiveness, we experimented with three text embedding methods and three visual embedding approaches. Current research primarily focuses on CoT and VQA individually <a href="#20"> [20] </a> <a href="#18"> [18] </a>. Our project aims to fill the gap by investigating the combined impact of CoT and VQA, contributing to the understanding of how these techniques can improve the reasoning capabilities of state-of-the-art models like GPT-4 <a href="#21"> [21] </a>.

The datasets utilized in our experiments are TextVQA and ScienceQA <a href="#19"> [19] </a> <a href="#17"> [17] </a>. TextVQA consists of 45,336 questions based on 28,408 images from the Open Images dataset, requiring reasoning and reading about text in the image <a href="#18"> [18] </a>. ScienceQA contains 21,208 multimodal multiple-choice science questions collected from elementary and high school science curricula, covering a wide range of subjects and offering a rich diversity of domains <a href="#8"> [8] </a>

We set up the following text embedding models for our experiments: 1) a simple QA model using DistilBERT as a baseline <a href="#15"> [15] </a>; 2) T5 with reasoning without image captions <a href="#9"> [9] </a>}; and 3) T5 with reasoning and image captions <a href="#9"> [9] </a>. For visual embedding models, we considered: 1) a baseline VQA model <a href="#11"> [11] </a>; 2) integrating visual embeddings with textual embeddings for the baseline textual model <a href="#13"> [13] </a>; and 3) visual embedding with textual embeddings for the T5 model <a href="#10"> [10] </a>.

We thought this would be a fruitful approach since VQA and CoT individually already improved model performance substantially on similar benchmarks <a href="#20"> [20] </a> <a href="#18"> [18] </a>. Using VQA together with CoT is a new approach, which we aimed to explore in this study <a href="#21"> [21] </a>.

Anticipated problems included limitations with computational resources, and we encountered some models returning surprisingly poor performance (RoBERTa) contrary to expectations <a href="#12"> [12] </a>. The very first thing we tried did not work, but we iteratively refined our approach to address these issues.\\

We did not use any code from repositories, but we used the following for reference:

1. Document Question Answering <a href="#2"> [2] </a>
2. Question Answering With T5 <a href="#1"> [1] </a>
3. Towards Data Science: Adding Custom Layers on Top of a Hugging Face Model <a href="#16"> [16] </a>
4. Multiple choice <a href="#3"> [3] </a>

By combining CoT and VQA techniques, we strive to demonstrate the potential of these approaches in enhancing the reasoning and question-answering capabilities of LMs. Our experiments may pave the way for further research and development in the field, leading to more accurate and reliable AI systems that can handle complex reasoning tasks across multiple modalities.

By integrating CoT and VQA, we hope to leverage the strengths of both approaches, enabling LMs to reason more systematically and accurately when provided with textual and visual information. This combined approach could be particularly useful for real-world applications where data comes in various forms and requires the integration of multiple sources of information for effective decision-making.

Moreover, our research could inspire future work on novel architectures and algorithms that capitalize on the synergies between CoT and VQA, pushing the boundaries of what is possible with current AI technology. Our findings may also contribute to the development of more interpretable models, as generating rationales for each choice can help explain the model's decision-making process, making it more transparent and understandable for human users.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!--Experiments & Results -->
<h2 id="Experiments and Results"> Experiments and Results </h2>

To evaluate the success of our proposed approach, we conducted a series of experiments using various models and configurations. We then compared the results, both quantitatively and qualitatively, to assess the effectiveness of our approach in enhancing the reasoning and question-answering capabilities of LMs.

The memory and computational requirements needed for creating scalable VQA models constrained us with using models that use both image and text features for question answering. We tried a couple of multi-modal frameworks like ViLT <a href="#5"> [5] </a> and VisualBERT <a href="#7"> [7] </a>. We fine-tuned the ViLT model on the ScienceQA dataset by manually creating domain-specific vocabulary and annotations with scores for probable answers. The model, however, did not perform well on the dataset as it was constrained to generating single-word answers and lacked the capability of generating coherent reasoning like Text-to-Text Language models (T5 <a href="#4"> [4] </a>).

<h3 id="TextQA"> ➤ TextQA Tasks </h3>
We started with a textual question-answering task to evaluate the reasoning capabilities of our models. The following models were used in our experiments:

* Baseline: A simple QA model based on DistilBERT and RoBERTa. These models were chosen due to their relatively small size, making them suitable for training with our computational resources. Moreover, they have been shown to perform decently on multiple-choice questions.
* T5 without reasoning: We trained a T5-small model without any reasoning capabilities to assess the impact of adding CoT reasoning.
* T5 with reasoning: We further trained a T5-small model with reasoning capabilities, integrating the CoT approach.
* T5 with reasoning and image captions: To assess the potential benefits of adding image information, we trained a T5-small model with reasoning capabilities and image captions.

<h3 id="VisualQA"> ➤ VisualQA Tasks </h3>
In the second phase of our experiments, we focused on the VQA task, integrating visual embeddings with textual embeddings.

* Fine-tuned VQA model on Science QA Dataset: We fine-tuned the pre-trained ViLT(Vision \& Language Transformer) model for visual question answering. We created domain-specific vocabulary and annotations using the ScienceQA dataset and used the ViLT model to generate answers. The ViLT model, however, did not perform well on the dataset as it was constrained to generating single-word answers and lacked the capability of generating coherent reasoning like Text-to-Text Language models (T5).
* We also attempted to integrate the visual embeddings from models like DETR into the VisualBert model. However, we were not successfully able to concatenate the visual and text features to re-train the VisualBert model. This was due to the varying hidden dimension of the textual encodings for different downstream task models in VisualBert.
* Integrated model with T5 textual embeddings: We generated image captions from a Vision Transformer model (ViT-GPT2 <a href="#6"> [6] </a>) and  used that along with textual input to T5-small model and experimented with different training strategies and settings to assess the impact of adding reasoning capabilities and image context.

We ran our experiments for answer generation on both TextVQA and ScienceQA dataset along with setting up the training for Answer and Explanation generation solely on the ScienceQA dataset where we had ground truth explanations (solutions) for which we measured the Rogue F1 scores.

<h3 id="Experimentation"> ➤ Experimental Settings and Hyperparameters </h3>  

We set-up our experiments on Google Cloud Platform using GPU setting for the training and evaluation of our models. For the experiments with the T5 model, we used the Adam Optimizer for training with a learning rate of 1e-5. We also used a linear learning rate scheduler with number of training steps as 10000 and a number of warm up steps as 500. We set the max input and output length depending on the downstream task we attempted to solve (128 for answer generation and 256 for explanation generation). 

We also explored gradient clipping to avoiding exploding gradient in the training strategy. We monitored both the training and validation loss after each epoch with the total number of epochs finally set to 10.

<h3 id="Metrics"> ➤ Metrics </h3>  

To measure the performance of the models, we used the following metrics:

1. Accuracy: We computed the accuracy by comparing the model’s predictions with the ground truth answers.
2. ROUGE $F_{1}$ score: These metrics evaluate the quality of the generated text by comparing it to the reference text. They provide a quantitative measure of the model’s performance. The $F_{1}$ score was chosen as it provides a balance between precision and recall.

<h3 id="KeyFindings"> ➤ Key Findings </h3>  

* Training and Validation Loss Curves: We analyzed the loss curves specific to all 6 model variants illustrated using Figures 1-6 respectively. Most of the curves showed a sharp dip in the training loss in the first epoch suggesting that the pre-trained T5 model was pretty quick in re-adjusting its weights to better align with the domain-specific examples. The validation losses in most cases were pretty low from the initial get-go. Also, we observed convergence in the training and validation losses for almost all model variants suggesting that running the model for more than 10 epochs would pose the risk of running into an overfitting problem. Also, for (Model6) which we found to be the best-performing model, we could see pretty low values of both training and validation losses when compared to other models from initial epochs.
* The models that were trained using image captions along with context and hint (Model 2) did not outperform the model without the image captions (Model 1) when compared on the validation dataset. This was slightly counter-intuitive but the reason for it might be that the model suffered from information overload and could not specifically utilize the captions when provided along with hint and context. This also highlights the importance of working with models that can exploit the mutual synergies of different modalities like text and vision features with an attention mechanism to generate coherent reasoning. The idea of using the image features as textual image captions did not yield enhanced performance.
* The model that was trained to generate answers and explanations simultaneously (Model3) was outperformed by the model trained on just generating answers (Model2). This emphasizes the fact that task-specific training gives  better results and prevents the model from information load.
* The models that were directly fine-tuned from the pre-trained T5 conditional model on generating both answers and explanations simultaneously (Model3) was outperformed by the model (Model4)  which was fine-tuned first on generating answers and then using that model checkpoint to generate answers and explanations. This makes sense as it emphasizes that the fine-tuned model on a domain-specific dataset could leverage its previous learnings to perform well on subsequent runs.
* Knowledge Distillation with teacher training: We found that the models which utilized the generated explanations from the same model in the previous run (Model5, Model 6) outperformed the model trained on directly generating answers respectively (Model2). This is also intuitive as using the model-generated explanations acts as positive feedback as the model learns to better its prediction on the answers using that as additional input.
* For the downstream task of answer generation, we also see that the T5-small model trained on the TextVQA dataset outperformed the other models tested on the ScienceQA dataset with a training accuracy of 65.92\% and an accuracy of 62.66\%. This is in accordance with our hypothesis as this dataset had more training examples and all the examples contained images unlike the ScienceQA dataset. Additionally, a lot of the questions in the ScienceQA dataset involved very domain-specific vocabulary and trickier questions for the model to learn from given only the caption and hint which is why we wish to further explore models that take into account vision features as well.

<h3 id="Results"> ➤ Results</h3>  
<h3 id="ModelOutput"> ➤ Model Output Samples </h3>  
  


<!-- References -->
<h2 id="References"> References </h2>
<a id="1"> [1] </a>
Abdokamr. Question answering with t5, Aug 2021 (https://www.kaggle.com/code/abdokamr/question-answering-with-t5)<br/>
<a id="2"> [2] </a>
Hugging Face. Document question answering. (https://huggingface.co/docs/transformers/tasks/document_question_answering)<br/>
<a id="3"> [3] </a>
Hugging Face. Multiple choice. (https://huggingface.co/docs/transformers/tasks/multiple_choice)<br/>
<a id="4"> [4] </a>
Hugging Face. T5 small model. (https://huggingface.co/t5-small)<br/>
<a id="5"> [5] </a>
Hugging Face. Vilt model documentation. (https://huggingface.co/docs/transformers/model_doc/vilt)<br/>
<a id="6"> [6] </a>
Hugging Face. Vision transformer model documentation. (https://huggingface.co/docs/transformers/model_doc/vit)<br/>
<a id="7"> [7] </a>
Hugging Face. Visual bert model documentation. (https://huggingface.co/docs/transformers/model_doc/visual_bert)<br/>
<a id="8"> [8] </a>
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot,
Dan Roth, and Jonathan Berant. Did aristotle use a laptop?
a question answering benchmark with implicit reasoning
strategies. Transactions of the Association for Computational Linguistics, 9:346–361 <br/>
<a id="9"> [9] </a>
Daniel Khashabi, Tushar Chaudhary, Ayush Sen, Jing
Chen, Yejin Choi, and Mirella Lapata. Unifiedqa: Crossing format boundaries with a single qa system (https://aclanthology.org/2020.findings-emnlp.171.pdf)<br/>
<a id="10"> [10] </a>
Wonjae Kim, Kyung-Min Lee, Jin-Hwa Lee, Chanho Park,
Youngeun Jang, Seonguk Park, Byeongchang Yoon, and
Sung Ju Hwang. Vilt: Vision-and-language transformer
without convolution or region supervision.  (https://arxiv.org/pdf/2102.03334.pdf)<br/>
<a id="11"> [11] </a>
iunian Harold Li, Yatskar Mark, Chen Da, Matt Hessel,
and Dragomir Radev. Visualbert: A simple and performant
baseline for vision and language (https://arxiv.org/pdf/1908.03557.pdf)<br/>
<a id="12"> [12] </a>
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke
Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly
optimized bert pretraining approach (https://arxiv.org/pdf/1907.11692.pdf)<br/>
<a id="13"> [13] </a>
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vil-
bert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks(https://arxiv.org/pdf/1908.02265.pdf)<br/>
<a id="14"> [14] </a>
OpenAI. Gpt-4 technical report <br/>
<a id="15"> [15] </a>
Victor Sanh, Lysandre Debut, Julien Chaumond, and
Thomas Wolf. Distilbert, a distilled version of bert:
smaller, faster, cheaper and lighter (https://arxiv.org/pdf/1910.01108v4.pdf)<br/>
<a id="16"> [16] </a>
Towards Data Science. Adding custom layers on top of a
hugging face mode (https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd)<br/>
<a id="17"> [17] </a>
ScienceQA. Scienceqa: A multimodal multiple-choice science question dataset. (https://scienceqa.github.io/)<br/>
<a id="18"> [18] </a>
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang,
Xinlei Chen, Dhruv Batra, DSOTA23evi Parikh, and Marcus Rohrbach. Towards vqa models that can read (https://arxiv.org/pdf/1904.08920.pdf)<br/>
<a id="19"> [19] </a>
Text VQA. Text vqa: A dataset for reasoning about text in images (https://textvqa.org/)<br/>
<a id="20"> [20] </a>
Ji Wei, Yixin Hao, Antoine Bosselut, and Yejin Choi.
Chain-of-thought prompting elicits reasoning in large language models (https://arxiv.org/pdf/2201.11903.pdf)<br/>
<a id="21"> [21] </a>
Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao,
George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models <br/>



# Work Division

See Table
[\[tab:contributions\]](#tab:contributions){reference-type="ref"
reference="tab:contributions"} for work division.

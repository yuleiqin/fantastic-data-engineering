<h1 align="center">Fantastic Data Engineering for Large Language Models</h1>
<p align="center"><i>A collection of data engineering methods for pretraining and aligning large language models.</i></p>
<div align="center">
  <a href="https://github.com/yuleiqin/fantastic-data-engineering/stargazers"><img src="https://img.shields.io/github/stars/yuleiqin/fantastic-data-engineering" alt="Stars Badge"/></a>
<a href="https://github.com/yuleiqin/fantastic-data-engineering/network/members"><img src="https://img.shields.io/github/forks/yuleiqin/fantastic-data-engineering" alt="Forks Badge"/></a>
<a href="https://github.com/yuleiqin/fantastic-data-engineering/pulls"><img src="https://img.shields.io/github/issues-pr/yuleiqin/fantastic-data-engineering" alt="Pull Requests Badge"/></a>
<a href="https://github.com/yuleiqin/fantastic-data-engineering/issues"><img src="https://img.shields.io/github/issues/yuleiqin/fantastic-data-engineering" alt="Issues Badge"/></a>
<a href="https://github.com/yuleiqin/fantastic-data-engineering/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/yuleiqin/fantastic-data-engineering?color=2b9348"></a>
<a href="https://github.com/yuleiqin/fantastic-data-engineering/blob/master/LICENSE"><img src="https://img.shields.io/github/license/yuleiqin/fantastic-data-engineering?color=2b9348" alt="License Badge"/></a>
</div>

# News📰
Under construction🔥🔥🔥

# Tools🛠️
## Popular Classifiers in NLP📊
- [FastText Classifier](https://arxiv.org/pdf/1607.01759) - Bag of Tricks for Efficient Text Classification (can be used for topic/domain/quality classification).

- [FASTTEXT.ZIP](https://arxiv.org/pdf/1612.03651) - Compressed FastText classification models.

- [An Open Dataset of High-Quality Mathematical Web Text](https://arxiv.org/abs/2310.06786) - A fastText-based classifier that evaluates the MathScore of any given contents for selection of mathematic corpus.


# Papers📑

## Surveys📝

- [A survey on data selection for language models](https://arxiv.org/pdf/2402.16827) - A systematic overview for data pipeline construction of language models. Any selection method, either via distribution matching or diversification, can be composed of: 1) utility function; 2) selection mechanism. During different stages of the pipeline (e.g., language filtering, data quality, domain knowledge, deduplication, toxic and explicit content removal, and data mixing), the selection method should be adjusted according to different selection objectives.



- [A Survey on Data Selection for LLM Instruction Tuning](https://arxiv.org/pdf/2402.05123) - Existing methods on collectiong instruction tuning datasets include: 1) reformulating the discriminative NLP datasets into generative ones; 2) self-instruct with seed prompts; 3) prompt mapping and evol-instruct. Popular methods on dataset selection can be simply classified as: 1) system of indicators; 2) trainable LLMs; 3) powerful LLMs; and 4) small models. The validation of different selection methods is often performed via direct scoring by GPT4 and human evaluation.


- [Deepcore: A comprehensive library for coreset selection in deep learning](https://arxiv.org/pdf/2204.08499) - Task-agnostic data sampling (coreset selection) methods include: 1) geometry-based methods (e.g., herding, kcenter-greedy); 2) uncertainty-based methods (e.g., least confidence/entropy/margin); 3) error/loss-based methods (forgetting; GraND/EL2N; importance resampling); 4) decision boundary-based (adversarial deepfool; contrastive active learning); 5) gradient matching-based (gradient approximation towards full set); 6) bi-level optimization-based (inner loop of model optimization and outer loop of datapoint selection); 7) sub-modularity-based (e.g., graph cut; facility location); 8) proxy-based (preference of a small model on data selection).



## Dataset Construction and Synthesis🚧

### Self-Instruct🤳

- [Self-instruct: Aligning language models with self-generated instructions](https://arxiv.org/pdf/2212.10560) - One most effective way to expand dataset is to bootstrap off generations from large language models themselves. It uses input-first or output-first approach to generate new instructions and filters out low-quality or highly duplicated/similar ones.

- [Muffin: Curating multi-faceted instructions for improving instruction following](https://openreview.net/pdf?id=1vrS1zwekw) - Three paradigms for expanding the instruction datasets: 1) scaling inputs; 2) scaling input-free tasks; and 3) scaling tasks per input. To choose appropriate input samples, input contents are selected from different domains with controled similar frequency. The entire scaling pipeline includes facets recognition, instruction brainstorm, instruction rematching, and output annotation.

- [Self-play with Execution Feedback: Improving Instruction-following Capabilities of Large Language Models](https://arxiv.org/pdf/2406.13542) - The two-stage instruction-followng data synthesis method is composed of: 1) instruction augmentation and verification and 2) query augmentation and verification. In the first step, human-verified seed instructions are provided for the supervisor model (e.g., larger powerful LLMs) to perform self-instruct. The augmented instructions are then used to generate their verification python codes and test cases, where only high quality, excutable triplets (test cases, instructions, verification functions) are kept via filtering (e.g., back translation check). In the second step, real-world queries are randomly selected from ShareGPT dataset and concatenated with the generated instructions as **complexity-augmented** instructions. These instructions are scored by a LLM to filter out contradictory ones (e.g., query imcompatible with instruction). Only valid instructions are sent to the supervisory model for response generation and verification. Correct, high quality responses can be used for **supervised fine-tuning** and their invalid counterparts can be employed as negative preferences in **DPO alignment**.


### Evo-Instruct🐛

- [Wizardlm: Empowering large language models to follow complex instructions](https://arxiv.org/pdf/2304.12244) - The **evolution** method to increase the complexity of existing instruction datasets is to add more variation (e.g., deepening, reasoning, concretizing, constraints, in-breadth extension, and complicated inputs) to the original samples, where the differences of the responses with respect to the nuances of inputs benefit the model's ability of following instructions.

- [Tree-Instruct: A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment](https://aclanthology.org/2024.lrec-main.1460.pdf) - The **complexity** of instructions can be represented as the number and diversity of nodes of their semantic trees. Therefore, the pipeline of improving instruction complexity includes: 1) tree construction; 2) nodes expansion; and 3) tree sentencization (remapping from tree to instruction).

- [Infiniy Instruct](https://github.com/FlagOpen/Infinity-Instruct) - The infinity instruct method first collects high-quality open instruction datasets and perform filtering and tagging to give detailed description about each instruction (e.g., domain, knowledge prerequisite, abilities, fine-grained categories). Based on these attributes, high-quality instructions are chosen as seeds for expansion via Evo-Instruct. Finally,
for each evolved instruction, multi-rounds conversations are generated.


## Dataset Deduplication🦄

### Exact Match🟰

- [URL normalization for de-duplication of web pages](https://dl.acm.org/doi/10.1145/1645953.1646283) - URL removel for efficient data-deduplication.

- [CCNet: Extracting high quality monolingual datasets from web crawl data](https://arxiv.org/pdf/1911.00359) - Hash code of each document is calculated and compared for deduplication.

- [Asynchronous pipeline for processing huge corpora on medium to low resource infrastructures](https://inria.hal.science/hal-02148693/file/Asynchronous_Pipeline_for_Processing_Huge_Corpora_on_Medium_to_Low_Resource_Infrastructures.pdf) - A non-collision resistant hashing algorithm isdeveloped to remove duplidates.

- [MinHash](http://www.misserpirat.dk/main/docs/00000004.pdf) [MinHashLSH](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [SimHash](https://dl.acm.org/doi/pdf/10.1145/509907.509965) - Efficient approximation methods of the hashing algorithm are devloped to handle massive corpora.


### Semantics🔤

- [Semdedup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/pdf/2303.09540) - The removal of redundant data pairs that are semantically similar but not identical can speed up training but also preserve performance.


## Dataset Pruning✂️

### LLM Pretraining (Methods, Tricks, Pipelines)🏋️

- [Doremi: Optimizing data mixtures speeds up language model pretraining](https://proceedings.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf) - The mixture of datasets across diverse domains can be determined by a small proxy model (e.g., 280M params) using group distributionally robust optimization. The weights of datasets are used for resampling the pretrained datasets for training a much larger model (e.g., 8B) with lower perplexity across all domains.


- [When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/pdf/2309.04564) - Common indicators (e.g., perplexity, EL2N, memorization ranking) are investigated for data quality measurement and dataset cleaning.

- [Beyond neural scaling laws: beating power law scaling via data pruning](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf) - Apart from the data measurement metrics, the proportion of pruned data with respect to the model size matters. Keep easy samples from small datasets, and difficult samples from big datasets.

- [Autonomous data selection with language models for mathematical texts](https://openreview.net/pdf?id=bBF077z8LF) - A simple pipeline to filter out mathematic samples from open-sourced corpus for continue pretraining. Direct scoring via LLMs is effective in selecting high-quality sampels.

- [Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models](https://arxiv.org/pdf/2405.20541) - The perplexity of datapoints inferred from a small reference model can be used to prune datasets for training LLMs. Medium and High perplexity selected (via frequency) datapoints are the most beneficial ones. However, the **marginal utility** diminishes when more data (e.g., over the requisite by scaling laws) are involved or more training epochs are repeated.

- [Codegen2: Lessons for training llms on programming and natural languages](https://arxiv.org/pdf/2305.02309) - One important lesson in organizing coding datasets for pretraining is to permute a portion of sequences for causal LLMs to get the code infilling ability. However, such permutation is not for free, where the performance (with a mixture loss of causal language modeling and prefix/suffix/middle sequence re-ordering) drops compared with the vanilla method (only with the causal language modeling loss).

- [The devil is in the details: A deep dive into the rabbit hole of data filtering](https://arxiv.org/pdf/2309.15954) - The pruning is conducted mainly via distribution alignment (between pretraining multi-modal datapoints and downstream datapoints): cluster-importance resampling; quality scoring-based reweighting and sampling; semantic deduplication (thresholding); enhancement on selecting samples from specific domains (e.g., digits).


- [Data similarity is not enough to explain language model performance](https://arxiv.org/pdf/2311.09006) - **Similarity metric** is not well-correlated with the downstreaming performance for pre-training models. Existing similarity methods (e.g., embedding similarities, token/n-gram distributions, and perplexity) are not correlated. The difficulty/complexity of downstreaming task datapoints (e.g., performance) is **NOT necessarily** associated with involving their similar counterparts in the pre-training corpus.

- [D4: Improving llm pretraining via document de-duplication and diversification](https://proceedings.neurips.cc/paper_files/paper/2023/file/a8f8cbd7f7a5fb2c837e578c75e5b615-Paper-Datasets_and_Benchmarks.pdf) - The pruning of datasets can be simply achieved by SemDeDup (semantic deduplication) and prototypicality filtering. Such dedupped and diversified subsets improve downstreaming performance even with repeating training epochs.

- [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/pdf/2402.09739) - QuRating defines quality criteria such as writing style, facts and trivia, educational value, and required expertise, and uses GPT-3.5-turbo to judge text pairs to generate labels for training the QuRater model. The fine-tuned QuRater model can then rate the quality of text. Experiments show that the language models trained with data selected by QuRating perform better than those trained with other data selection methods, and different quality criteria have different impacts on model performance, with educational value and required expertise being the most significant.


### LLM Instruction Fine-tuning and Aligning🛟
- [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/pdf/2308.12032) - Select 1K samples from each cluster of the fine-tuning datasets and construct "experiencing" models. Evaluate all datapoints using these models via instruction-following difficulty, which is defined as the conditioned answer score/direct answer score. **Choose the datapoints with moderate IFD scores!**

- [Alpagasus: Training a better alpaca with fewer data](https://arxiv.org/pdf/2307.08701) - It is surprisingly easy and effective to employ strongger models (e.g., GPT3.5, GPT4) to directly score datapoints in terms of helpfulness and accuracy. **Note that coding datasets might not be fairly scored due to their nature.**

- [Instruction mining: High-quality instruction data selection for large language models](https://arxiv.org/pdf/2307.06290) - The loss of the base model on dev and test sets can be viewed as a proxy for measuring the quality of datasets. To avoid the high-cost of retraining and evaluation of base models, one efficient way is to directly estimate the loss of the model for each datapoint based on linear regression with quality indicators (e.g., input length, output length, reward score, perplexity, MTLD, KNN-i, and uni-eval metrics).

- [Openassistant conversations-democratizing large language model alignment](https://proceedings.neurips.cc/paper_files/paper/2023/file/949f0f8f32267d297c2d4e3ee10a2e7e-Paper-Datasets_and_Benchmarks.pdf) - Each example following the **conversation tree** structure is collected and annotated with human. Dataset pruning is performed with human preference (e.g., creativity, quality, humor, helpfulness, violence, rudeness).

- [Towards a Unified Multi-Dimensional Evaluator for Text Generation](https://arxiv.org/pdf/2210.07197) - The evaluation on text corpus can be explained via **naturalness**, **coherence**, and **understandability**.

- [Rethinking the Instruction Quality: LIFT is What You Need](https://arxiv.org/pdf/2312.11508) - The pipeline of **expansion first and compression next** enhances both the diversity and quality of the original dataset. Data expansion is performed by GPT4 rewriting (depth, breath, CoT). Diversity is defined via PCA where samples of top-row variances are kept for diversity. Quality is measured by GPT4 direct scoring.


- [Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning](https://arxiv.org/pdf/2305.09246) - One most cost-efficient way to perform instruction fine-tuning is simply choose the datapoints that highly resemble the downstream datapoints with limited instruction formats. The pipeline of the data pruning consists of **embedding encoding and projecting**, **clustering**, **sampling**, and **model training and infering**. Especially, the sampling by diversity is often adopted as an effective coreset sampling method.

- [Automated data curation for robust language model fine-tuning](https://arxiv.org/pdf/2403.12776) - The pipeline of auto-cleaning instruction datasets consists of **auto-filter** and **auto-correct**. High-confident samples are first selected via the BSDetector to fine-tune a LLM. Then, candidate generated answers are inferred for all in-confident samples using the fine-tuned LLM. Finally, preference scoring between the original ground-truth answer and the generated answers are obtained using a base model where highly-confident generated answers are kept as the **corrected** answers.

- [Technical Report: Competition Solution For BetterMixture](https://arxiv.org/pdf/2403.13233) - Giver existing popular open-sourced datasets and the training budget (e.g., number of maximum training tokens), the best option to mix datasets for downstream performance lies in the details of filtering and balancing different datapoints. The entire pipeline includes deduplication (exact match), quality thresholding (language identification, perplexity, IFD scoring and voting), and diversity selection (kcenter-greedy sampling).

- [Exploring Learning Complexity for Downstream Data Pruning](https://arxiv.org/pdf/2402.05356) - The learning complexity of datapoints iss defined as the averaged prediction confidence of subnets with different capacity (predicted label consistency between kNN samples for classification and the sum of perplexity reciprocal for regression). The principle of pruning is keeping the easy and diverse samples.

- [Balanced Data Sampling for Language Model Training with Clustering](https://arxiv.org/pdf/2402.14526) - Cluster first, then uniformly sample datapoints from each cluster until exhaustion.


- [What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning](https://arxiv.org/pdf/2312.15685) - Datasets can be measured from three dimensions: complexity, quality, and diversity. A dataset is obtained by first performing **evolution on instruction complexity and response quality** of datasets via Evo-Instruct, respectively on each sample and rank these variants from high to low. Subsequently, diversity is considered where newly added samples should share low similarity with the existing dataset.


- [Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning](https://arxiv.org/pdf/2402.00530) - A small model (GPT2-125m) can be used to filter instruction-tuningdatasets for training a much larger and stronger model. Specifically, the perplexity and IFD scores of a datapoint from small language models are highly correlated with those from a LLM. Samples with top-k IFD scores (IFD score<1) are chosen for dataset reduction.



## Dataset Mixture🥄

- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) - The mixture weights of datasets from different domains are set according to the performance of a small model trained independently on each component dataset. Small-yet-high-quality datasets such as Wikipedia are not over-sampled.


- [SlimPajama-DC: Understanding Data Combinations for LLM Training](https://arxiv.org/abs/2309.10818) - The proportions of different domains in the deduplicated SlimPajama/RefinedWeb dataset are studied with 7 different configurations where their total number of tokens remains the same. The results of 7 configurations in terms of both downstream metrics and losses suggests appropriate weights for datasets from different sources.


- [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/pdf/2403.16952) - It discovers the quantitative predictability of model performance regarding data mixtures, and formulates it as the data mixing laws. It proposes a pipeline to predict model performance on different mixture proportions through nested use of the scaling laws of training steps, model sizes, and data mixing laws. 


- [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492) - The paper proposes the RegMix method to automatically identify an effective data mixture for language model pre-training by formulating it as a regression task. It involves training small models with diverse data mixtures and fitting a regression model to predict their performance. The top-ranked mixture is then used to train a large-scale model. The experiments show that RegMix is superior to human selection, achieves comparable or better results than DoReMi with only 10% of the compute budget. It also finds that data mixtures significantly impact performance, web corpora have a stronger correlation with downstream performance, domains interact complexly, and the data mixture effects transcend scaling laws. 

- [Data Mixing Made Efficient: A Bivariate Scaling Law for Language Model Pretraining](https://arxiv.org/pdf/2405.14908) - BiMix adopts the bivariate scaling law to reveal the impact of data quantity and mixing proportions using entropy proxies. The scaling behavior is disentangled into two dimensions: 1) scaling step with fixed proportions, and 2) scaling proportion at fixed steps.


- [How Abilities in Large Language Models are Affected by Supervised Fine-tuning Data Composition](https://arxiv.org/pdf/2310.05492) - Different skills require different strategies of supervised fine-tuning. Sequential training results in catastrophic forgetting and the proposed dual-stage mixed fine-tuning strategy alleviates such forgetting and keeps the general capabilities simultaneously. Specifically, the specialized domain datasets are trained in the first stage and then mixed with general datasets for fine-tuning in the second stage.


## Dataset Selection☑️

### LLM-based Evaluation⚖️
- [Mods: Model-oriented data selection for instruction tuning](https://arxiv.org/pdf/2311.15653) - Models (LLMs) are investigated during data selection, where three metrics of datasets are defined: 1) quality, 2) coverage, and 3) necessity. For quality measurement, the reward model (via preference scoring) is used to rank all datapoints. For coverage, kcenter-greedy sampling is employed to reduce the number of samples without losing generalizability. For necessity, the responses generated from the model (fine-tuned using the kcenter-greedy sampled datapoints) are evaluated using the reward model. Samples that achieve low scores need to be added into the fine-tuning set.

- [Skill-it! a data-driven skills framework for understanding and training language models](https://proceedings.neurips.cc/paper_files/paper/2023/file/70b8505ac79e3e131756f793cd80eb8d-Paper-Conference.pdf) - The training of LLMs should follow a certain natural order that mimics how humans acquire independent skills and knowledge. It first estimates the skill affinity matrix (pre-requisite edges) for each training and validation skill, and then performs skill graph-aware coreset sampling for online learning.




### Influence Estimation (Importance of Datapoints)🏗️


- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/pdf/2402.04333) - The targeted instruction tuning aims at finding the most influential data that improves the downstreaming performance of LLMs, where low-rank gradient similarity search is developed to efficiently pinpoint the training examples that resemble the few-shot testing cases. Specifically, for each dataset, the base model is warmed-up by experiencing a small portion of samples for more accurate, stable estimation of gradients. Then, the gradient-based trajectory influence estimation is extended to work with Adam optimizer. The LoRA technique, together with random projection, is involved to efficiently compute and store the gradient. Finally, the average gradient of testing cases over epochs is computed for similarity measurement with respect to the gradients of each training case, where those top-ranked samples are the most influential ones.


- [What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation](https://proceedings.neurips.cc/paper_files/paper/2020/file/1e14bfe2714193e7af5abc64ecbd6b46-Paper.pdf) - The value estimators of memorization and influence help pinpoint the most important datapoints that affect test-time performance significantly.

- [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://proceedings.neurips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf) - Important samples can be found at an early stage using indicators like forgetting score, gradient norm (GraNd), and Error l2 norm (EL2N). A high ratio of pruning would degrade overall performance due to overfitting of samples with label errors or high difficulty.

- [Dsdm: Model-aware dataset selection with datamodels](https://arxiv.org/pdf/2401.12926) - Datamodels can be simply defined as a linear model (e.g., logistic regression) and optimized via TARK estimator. The selected datapoints are **NOT necessarily** similar to the samples in the downstream tasks.

- [Data selection for language models via importance resampling](https://proceedings.neurips.cc/paper_files/paper/2023/file/6b9aa8f418bde2840d5f4ab7a02f663b-Paper-Conference.pdf) - Hashed n-gram features are **fast** and **effective** as representation embeddings. The importance of each datapoint is estimated by a bag of hashed n-grams model, where samples with higher probability (present in the target data) are assigned higher weights. Given any downstream datasets, the most similar samples in the pretraining corpus are **NOT necessarily** leading to the best downstreaming performance. But the most dissimilar ones do perform worst.


- [MATES: Model-Aware Data Selection for Efficient Pretraining with Data Influence Models](https://arxiv.org/pdf/2406.06046) - A data influence model (e.g., BERT-base), updating alternatively, continuously adapts to the evolving data preferences of the pretrained model (e.g., Pythia-410M/1B) and selects the most effective datapoints for the current pretraining.


- [Data selection for fine-tuning large language models using transferred shapley values](https://arxiv.org/pdf/2306.10165) - Sharply values can be approximated and aggregated to sample datasets, where the lowest scored samples are removed first until the proxy model (A_src) reaches the optimal performance on the validation set. Then, the selected dataset is used to train the target model (A_tgt).

- [From Random to Informed Data Selection: A Diversity-Based Approach to Optimize Human Annotation and Few-Shot Learning](https://arxiv.org/pdf/2401.13229) - Proposes an automatic data selection architecture for few-shot learning, including three methods: Reverse Semantic Search (RSS), Ordered Clustering (OC), and Limited Lexical Similarity (LLS).
Introducing the automatic data selection architecture based on active learning principles to identify the most informative and representative data points for annotation. Conducting an extensive analysis of the architecture's various implementations, highlighting its effectiveness in building the first version of a dataset in the context of low-resource text classification. 

### Downstream Metrics and Offline Indicators🧪
- [Do We Need to Create Big Datasets to Learn a Task?](https://aclanthology.org/2020.sustainlp-1.23.pdf) - Small-yet-important datasets can be efficiently collected simply by iteratively adding sampled subsets from big sets that contribute to downstream metrics. The cost-efficient AFLite filtering strategy, together with pre-defined data quality indicators (DQI), further reduces the size of the chosen datasets.

- [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/pdf/1907.10641) - For filtering out low-quality datapoints, it is feasible to evaluate their predictability scores (namely the number of times that one datapoint is correctly classified / the number of total testing times) and choose the top ranked datapoints. Step 1. Pre-compute representation embeddings for all datapoints. Step 2. Randomly partition datasets into training and validation splits and train proxy models (e.g., linear classifier) on the training set. Step 3. Evaluate the testing set. Step 4. Iterate from Step 2 to Step 3 until a pre-defined number of testing times is reached. Step 5. Calculate the predictability scores and choose the top-ranked, thresholded datapoints. Step 6. Iterate from Step 2. to Step 5. until the data quota is met.

- [Adversarial Filters of Dataset Biases](https://arxiv.org/pdf/2002.04108) - The AFLite aims at reducing the spurious bias of dataset and therefore improves model generalization on OOD samples.

- [DQI: Measuring Data Quality in NLP](https://arxiv.org/pdf/2008.03964) - The intuitive and manually-designed metrics for evaluating the data quality in NLP include vocabulary, inter-sample N-gram frequency and relation, inter-sample STS, intra-sample word similarity, intra-sample STS, N-Gram Frequency per Label, and Inter-split STS.

- [MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment](https://link.springer.com/article/10.3758/brm.42.2.381) - Metrics on **diversity measurement**.

- [Measuring Lexical Diversity in Texts: The Twofold Length Problem](https://arxiv.org/ftp/arxiv/papers/2307/2307.04626.pdf) - Metrics on **diversity measurement with respect to text length**.

- [Efficient k-nearest neighbor graph construction for generic similarity measures](https://dl.acm.org/doi/pdf/10.1145/1963405.1963487) - The distance to approximate i-th nearest neighbors can be used for **diversity measures**.

- [Data diversity matters for robust instruction tuning](https://arxiv.org/pdf/2311.14736) - The trade-off exists between the quality and diversity of datapoints. Diversity-prioritized datapoints can improve the worst-case performance. Diversity measures: **maximized similarity** between the sum of selected datapoints and the newly added one from the remaining full set. Quality measures: **ChatGPT direct scoring** and **reward model preference scoring**.

- [Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints](https://openreview.net/pdf?id=yb5xV8LFDq) - The selection of datapoints under a constrained budget can be implemented as the lexicographic bilevel-optimization, where the inner loop optimizes model parameters and the outer loop optimizes data selection. When optimizing the selection mask, the minimization of loss terms is relaxed to allow smaller dataset size.

- [Dele: Data Efficient LLM Evaluation](https://openreview.net/pdf?id=I8bsxPWLNF) - An adaptive effective sampling method can expedite LLM evaluation without losing discriminability of existing benchmarks. The candidate pool of sampling methods include: 1) random sampling; 2) clustering-based sampling (e.g., topic-modeling, DBScan, LDA, k-means, spectral); 3) quality-based sampling (spelling errors, average word-length, count of repeating words, compound probability distribution, lexical diversity); 4) difficulty-based sampling (difficult-words percentage, dale-chall formula, flesh reading ease, gunning fog).

- [Gio: Gradient information optimization for training dataset selection](https://arxiv.org/pdf/2306.11670) - To keep the selected dataset representative, it is feasible to use the KL-divergence as an measure between the sampled dataset and the target dataset (e.g., downstream datasets). Given an embedding model, the selection is performed by minimizing the KL divergence between two distribution where the newly added datapoint is determined by gradient information of the KL divergence.


- [Data-Efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond](https://arxiv.org/pdf/2402.17327) - A new data selection approach based on k-means clustering and sensitivity sampling can be applied for fine-tuning foundation models.


### Uncertainty❓

- [Quantifying uncertainty in answers from any language model and enhancing their trustworthiness](https://arxiv.org/pdf/2308.16175) - The pipeline of BSDetector uses both self-consistency and direct scoring to estimate the confidence of a LLM on any given instruction triplet (instruction, content, answer).

- [An Experimental Design Framework for Label-Efficient Supervised Finetuning of Large Language Models](https://arxiv.org/pdf/2401.06692) - The selection of datapoints (e.g., prompts) for supervised-finetuning can be mainly categorized as: 1) uncertainty-based selection; 2) k-center selection (e.g., k-center greedy); and 3) submodular selection (maximized diversity). Specifically, uncertainty metrics are defined as: 1) mean entropy; 2) least confidence; 3) mean margin; 4) min margin.



## :man_astronaut: Show your support☕️

Give a ⭐️ if this project helped you!

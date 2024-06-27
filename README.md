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



# Tools
## Popular Classifiers in NLP
- [FastText Classifier](https://arxiv.org/pdf/1607.01759) - Bag of Tricks for Efficient Text Classification (can be used for topic/domain/quality classification).

- [FASTTEXT.ZIP](https://arxiv.org/pdf/1612.03651) - Compressed FastText classification models.


# Papers

## Dataset Pruning

### LLM Pretraining (Methods, Tricks, Pipelines)
- [When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/pdf/2309.04564) - Common indicators (e.g., perplexity, EL2N, memorization ranking) are investigated for data quality measurement and dataset cleaning.

- [Beyond neural scaling laws: beating power law scaling via data pruning](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf) - Apart from the data measurement metrics, the proportion of pruned data with respect to the model size matters. Keep easy samples from small datasets, and difficult samples from big datasets.

### LLM Instruction Fine-tuning and Aligning
- [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/pdf/2308.12032) - Select 1K samples from each cluster of the fine-tuning datasets and construct "experiencing" models. Evaluate all datapoints using these models via instruction-following difficulty, which is defined as the conditioned answer score/direct answer score. **Choose the datapoints with moderate IFD scores!**



## Dataset Selection

### Influence Estimation (Importance of Datapoints)
- [What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation](https://proceedings.neurips.cc/paper_files/paper/2020/file/1e14bfe2714193e7af5abc64ecbd6b46-Paper.pdf) - The value estimators of memorization and influence help pinpoint the most important datapoints that affect test-time performance significantly.

- [Deep Learning on a Data Diet: Finding Important Examples Early in Training](https://proceedings.neurips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf) - Important samples can be found at an early stage using indicators like forgetting score, gradient norm (GraNd), and Error l2 norm (EL2N). A high ratio of pruning would degrade overall performance due to overfitting of samples with label errors or high difficulty.

### Downstream Metrics and Offline Indicators
- [Do We Need to Create Big Datasets to Learn a Task?](https://aclanthology.org/2020.sustainlp-1.23.pdf) - Small-yet-important datasets can be efficiently collected simply by iteratively adding sampled subsets from big sets that contribute to downstream metrics. The cost-efficient AFLite filtering strategy, together with pre-defined data quality indicators (DQI), further reduces the size of the chosen datasets.


- [DQI: Measuring Data Quality in NLP](https://arxiv.org/pdf/2008.03964) - The intuitive and manually-designed metrics for evaluating the data quality in NLP include vocabulary, inter-sample N-gram frequency and relation, inter-sample STS, intra-sample word similarity, intra-sample STS, N-Gram Frequency per Label, and Inter-split STS.



## :man_astronaut: Show your support

Give a ⭐️ if this project helped you!
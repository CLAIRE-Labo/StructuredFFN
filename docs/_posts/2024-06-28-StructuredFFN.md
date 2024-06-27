---
layout: post
title: Building on Efficient Foundations Effectively Training LLMs with Structured Feedforward Layers
---

**Author list: Xiuying Wei (CLAIRE, EPFL),  Skander Moalla (CLAIRE, EPFL), Razvan Pascanu (Google DeepMind), Caglar Gulcehre (CLAIRE, EPFL)**

## Abstract

State-of-the-art results in large language models (LLMs) often rely on scale, which becomes computationally expensive. This has sparked a research agenda to reduce these models' parameter count and computational costs without significantly impacting their performance. Our study focuses on transformer-based LLMs, specifically targeting the computationally intensive feedforward networks (FFN), which are less studied than attention blocks. We consider three candidate linear layer approximations in the FFN by combining efficient low-rank and block-diagonal matrices.  In contrast to many previous works that examined these approximations, our study i) explores these structures from the training-from-scratch perspective, ii) scales up to 1.3B parameters, and iii) is conducted within recent Transformer-based LLMs rather than convolutional architectures. We first demonstrate they can lead to actual computational gains in various scenarios, including online decoding when using a pre-merge technique.  Additionally, we propose a novel training regime, called *self-guided training*, aimed at improving the poor training dynamics that these approximations exhibit when used from initialization. Experiments on the large RefinedWeb dataset show that our methods are both efficient and effective for training and inference. Interestingly, these structured FFNs exhibit steeper scaling curves than the original models. Further applying self-guided training to the structured matrices with 32% FFN parameters and 2.5$\times$ speed-up enables only a 0.4 perplexity increase under the same training FLOPs. Finally, we develop the wide and structured networks surpassing the current medium-sized and large-sized Transformer in perplexity and throughput performance.


## Method

### Structured linear parametrization
We consider three structured parameterizations to approximate a linear layer ($Wx$) as below which have demonstrated computational gains on existing hardware.

* LowRank: $Wx \approx U^r(V^rx)$, where the superscript $^r$ is used to indicate matrices projecting in or from low dimensional states.
* BlockShuffle (two block-diagonal matrices, same as Monarch [1]): $Wx \approx f^{-1}(U^b f(V^bx))$, where  $V^b$ and $U^b$ are block-diagonal matrices and the shuffle function $f(\cdot)$ enables global feature mixing by cycling different blocks.
* BlockDense (block-diagonal followed by a dense matrix): $Wx \approx U^r(V^bx)$. Technically, the second projection does not need to be a low-rank approximation to reduce the parameter. But in practice, we chose the low-rank one with superscript $r$ to limit our search space.

The figure below shows how they perform and their reduced parameters and MAC. 

<img src="assets/method.png" />



Then, we go deeper to investigate their common challenges including efficiency and optimization.

###  Maintaining efficiency during online decoding

Challenge: While they have demonstrated materialized computational gains, they face challenges in the practical online decoding scenario of LLM, which may process only limited input tokens at one time, leading to under-utilization of computing resources and decreased efficiency due to the additional linear projection. 

Pre-merge technique: We address this with a pre-merge technique that restores the original dense efficiency when the total number of tokens is quite small (e.g., 16). Taking advantage of the fact that these parametrizations do not have non-linearity, we propose to combine the structured matrices into a single dense layer and keep both the structured and the dense one for online decoding. Then, we can dynamically decide which parametrization to use based on the current batch size and setting.



### Addressing the optimization challenge

Challenge: Using the efficient parametrization from initialization can suffer from optimization difficulty because the deep linear parametrization introduces additional symmetries, which is a source of proliferation of saddle points and generally less smooth loss function as pointed out in [2]. Empirically, we show that the deep linear form of  $U(Vx)$ leads to instability and loss spike or to slow convergence compared to the dense linear projection in the figure below.

<img src="assets/training_dynamic.png" />



Self-guided training: Addressing poor training dynamics by tuning the learning rate and gradient clipping is costly and unstable. We propose a simpler, cost-effective approach called self-guided training, requiring minimal hyperparameter re-tuning. This method uses dense parametrization to efficiently navigate early stages, where symmetries introduced by the structured parametrization impact feature specialization, then transfers the control to $U$ and $V$, defined as:

​																		$o = \alpha \cdot W x + (1-\alpha) \cdot U(Vx)$,

$o$ is the layer's output, and $\alpha$ decays following a cosine scheduler. As a residual component, learning $W$ is unaffected by the additional saddles and pathologies, allowing units to specialize. This *guides* the training of $U$ and $V$, which are forced slowly to take over by providing the hidden units semantics learned by $W$. The loss curves above show that such a method makes the training dynamics much better.

For more details, please check the paper.

## Experiments

We conduct our experiments at scale on Transformers ranging from 110M to 1.3B parameters. We demonstrate the efficiency of these parametrizations, conduct a scaling analysis that structured matrices have steeper scaling curves compared to the dense ones, and validate that self-guided training can boost the final performance efficiently. Finally, we design the wide and structured networks by combing the GQA [4], improving both the perplexity and throughput. 

### Evaluating latency results

We investigate the efficiency of structured FFN and consider different numbers of tokens to discuss different scenarios.

- Large number of tokens (usually concerning training, the prefill phase of inference, and extensive decoding cases)

  From width 1536, LowRank and BlockDense begin to enable about a 1.4$\times$ speed-up and a 2.5$\times$ speed-up with 63% and 32% parameters, respectively.

  <img src="assets/latency.png" />



- Small number of tokens (may happen at the decoding stage, especially for the online case)

  We vary the batch of tokens to determine when to use efficient alternatives or choose pre-merged dense matrices. For example, with a 2048-width FFN, it is difficult to fully utilize resources on GPU with limited tokens. The performance improves significantly when using width 5120 and 6144, such as speed improvements of 2.63$\times$ speed-up of LowRank with 32% FFN parameters on total number of tokens of 2048 and 2.81$\times$ acceleration of BlockDense with 32% parameters on 1536 tokens.

  
  <img src="assets/latency_bs.png" />


### Findings on efficient training

- Comparison between structured FFNs

  With the model and training FLOPs fixed, we show that LowRank and BlockDense can be better than the BlockShuffle for FFN in NLP tasks. However, we think this is task-dependent, because in vision tasks where block-diagonal matrices are better for local information, we find that block-diagonal matrix is a more suitable inductive bias (see experiments in the appendix).

  ![](assets/gpt.png)



- Scaling analysis 

  As we scale the model size, we find steeper scaling curves of structured matrices. Below, it's a figure for LowRank, but the other two hold similar curves. Specifically, 

  ​	*(i) The structured matrices exhibit steeper scaling curves compared to the dense networks, indicating significant potential for these efficient designs in LLMs.*

  ​	 *(ii) The scaling curve of 32\% parameters of FFN is steeper than the 63\% parameters of FFN highlights the scaling potential of highly structured large models.*

  ​	*(iii) Given fixed training FLOPs budget, a wider and structured network with more tokens may achieve comparable or superior performance to dense networks at the optimal trade-off.*

  <img src="assets/scaling_law_lowrank.png" />


### Self-guided training 

  With the self-guided training, our performance gets closer to dense models. For example, with the same training FLOPs, our 1.3B model has a 0.4 perplexity loss vs. the dense one and enjoys about 2.5x FFN speed-up for inference. Additionally, we compare our method with another advanced baseline that trains structured parametrizations with more tokens, showing that ours achieves comparable or superior results even with the same number of tokens.

  

<img src="assets/fig_sgt_lowrank.png" />

### Wide and Structured network

As maintaining the parameter ratio of attention to FFN can be important, in this section, we use GQA to make attention efficient and LowRank for FFN, designing a wide and structured network from Transformer-m and Transformer-l. To match the training FLOPs, we either train on more tokens or apply self-guided training. 

It can be seen that our methods achieve an 8% and 17% maximum throughput boost, respectively, while maintaining or slightly improving perplexity. TP refers to the maximum throughput measured on a generation length of 256.

<img src="assets/wide_structured.png" />



## Conclusion and Limitation

Conclusion: In this paper, we conducted extensive experiments investigating the use of structured matrices to parameterize FFN in Transformers, with models up to 1.3B parameters on the RefinedWeb dataset. Our primary aim was not to determine which structured matrices perform best, as this can be task-dependent, but to explore common issues including efficiency and optimization challenges of existing structured matrices as well as BlockDense. 

Limitation: BlockDense and BlockShuffle are more complicated than LowRank. In this work, we only explored a limited range of hyperparameter settings of them. Also, we primarily focused on language modeling with limited vision experiments included in the appendix. Additionally, we did not explore the optimal scaling laws for structured matrices, which may further enhance performance. 

## References

[1]. Monarch: Expressive Structured Matrices for Efficient and Accurate Training. ICML2022

[2]. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. ICLR2016

[3]. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP2023

## Useful Links

Paper:https://arxiv.org/pdf/2406.16450

Code: https://github.com/CLAIRE-Labo/StructuredFFN
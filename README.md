# UncSamp: Self-training Sampling with Monolingual Data Uncertainty for Neural Machine Translation
Implementation of our paper "Self-training Sampling with Monolingual Data Uncertainty for Neural Machine Translation" to appear in ACL 2021. [[paper]](https://www.aclweb.org/anthology/2021.acl-main.xxx/)


## Brief Introduction
Self-training has proven effective for improving NMT performance by augmenting model training with synthetic parallel data. The common practice is to construct synthetic data based on a randomly sampled subset of large-scale monolingual data, which we empirically show is sub-optimal. In this work, we propose to improve the sampling procedure by selecting the most informative monolingual sentences to complement the parallel data. To this end, we compute the uncertainty of monolingual sentences using the bilingual dictionary extracted from the parallel data. Intuitively, monolingual sentences with lower uncertainty generally correspond to easy-to-translate patterns which may not provide additional gains. Accordingly, we design an uncertainty-based sampling (**UncSamp**) strategy to efficiently exploit the monolingual data for self-training, in which monolingual sentences with higher uncertainty would be sampled with higher probability. Experimental results on large-scale WMT English⇒German and English⇒Chinese datasets demonstrate the effectiveness of the proposed method. Extensive analyses provide a deeper understanding of how the proposed method improves the translation performance.

<div align="center">
    <img src="/image/UncSamp.png" width="70%" title="Framework of Self-training with Uncertainty-Based Sampling."</img>
    <p class="image-caption">Figure 1: The framework of self-training with uncertainty-based sampling.</p>
</div>


## Reference Performance
We evaluate the proposed **UncSamp** approach on two high-resource translation tasks. As shown, our Transformer-Big models trained on the authentic parallel data achieve the performance competitive with or even better than the submissions to WMT competitions. Based on such strong baselines, self-training with RandSamp improves the performance by +2.0 and +0.9 BLEU points on En⇒De and En⇒Zh tasks respectively, demonstrating the effectiveness of the large-scale self-training for NMT models. 

With our **UncSamp** approach, self-training achieves further significant improvement by +1.1 and +0.6 BLEU points over the random sampling strategy, which demonstrates the effectiveness of exploiting uncertain monolingual sentences.

<div align="center">
    <img src="/image/Results.png" width="75%" title="Main results."</img>
    <p class="image-caption">Table 1: Evaluation of translation performance.</p>
</div>

Further analyses suggest that our **UncSamp** approach does improve the translation quality of high-uncertainty sentences and also benefits the prediction of low-frequency words at the target-side.

<div align="center">
    <img src="/image/Analysis-UncSent.png" width="50%" title="Analysis for Uncertain Sentences."</img>
    <p class="image-caption">Table 2: Analysis for uncertain sentences.</p>
</div>

<div align="center">
    <img src="/image/Analysis-LowFreq.png" width="50%" title="Analysis for Low Frequency Words."</img>
    <p class="image-caption">Table 2: Analysis for low frequency words.</p>
</div>


layout: project_page
permalink: /

title: Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning
authors:
    Jingyuan Zhang, Yiyang Duan, Shuaicheng Niu, YANG CAO, Wei Yang Bryan Lim
affiliations:
    NTU
paper: https://arxiv.org/abs/2410.07738
code: https://github.com/ntu-zjy/DomainFL
data: https://paperswithcode.com/dataset/domainnet


<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
Federated Domain Adaptation (FDA) is a Federated Learning (FL) scenario where models are trained across multiple clients with unique data domains but a shared category space, without transmitting private data. The primary challenge in FDA is data heterogeneity, which causes significant divergences in gradient updates when using conventional averaging-based aggregation methods, reducing the efficacy of the global model. This further undermines both in-domain and out-of-domain performance (within the same federated system but outside the local client), which is critical in certain business applications. To address this, we propose a novel framework called \textbf{M}ulti-domain \textbf{P}rototype-based \textbf{F}ederated Fine-\textbf{T}uning (MPFT). MPFT fine-tunes a pre-trained model using multi-domain prototypes, i.e., several pretrained representations enriched with domain-specific information from category-specific local data. This enables supervised learning on the server to create a globally optimized adapter that is subsequently distributed to local clients, without the intrusion of data privacy. Empirical results show that MPFT significantly improves both in-domain and out-of-domain accuracy over conventional methods, enhancing knowledge preservation and adaptation in FDA. Notably, MPFT achieves convergence within a single communication round, greatly reducing computation and communication costs. To ensure privacy, MPFT applies differential privacy to protect the prototypes. Additionally, we develop a prototype-based feature space hijacking attack to evaluate robustness, confirming that raw data samples remain unrecoverable even after extensive training epochs. The complete implementation of MPFL is available at \url{https://anonymous.4open.science/r/DomainFL/}.
        </div>
    </div>
</div>

## Background
In Federated Learning (FL), many existing methods assume clients’ data are i.i.d. (independently and identically distributed), enabling straightforward model-parameter averaging (e.g., FedAvg) to learn a global model. However, in real-world scenarios, data often come from different domains, sharing only the label space but differing in their distributions. This scenario is known as Federated Domain Adaptation (FDA). Under FDA, the large domain gaps across clients undermine the effectiveness of naive averaging-based approaches, making it challenging to achieve good performance both on each local domain and on out-of-domain data.



## Motivation
Numerous fields—such as finance, healthcare, and image recognition—require models that not only perform well on each client’s own domain (in-domain) but also generalize to other domains (out-of-domain). Traditional solutions often overfit to local domains, losing cross-domain generalization, or force an averaged global model that poorly fits each unique domain. A more effective, privacy-preserving method is needed to capture domain-specific knowledge and also aggregate insights across domains without sharing raw data.



## Problem Statement
Under FDA, each client holds data from a unique domain (but with the same label space). The goal is to jointly train a model (or local/global models) that meets two key requirements:

1. **Domain Knowledge Preservation (ind accuracy)**: Retain high accuracy on each client’s own domain.
2. **Domain Knowledge Adaptation (ood accuracy)**: Transfer knowledge learned from other domains to achieve high accuracy on out-of-domain data.

Because averaging model parameters often fails in the presence of substantial domain gaps, a new aggregation mechanism is necessary.



## Methodology
The proposed framework, **MPFT (Multi-domain Prototype-based Federated Fine-Tuning)**, addresses FDA in three main steps:

1. **Prototype Generation**
   - Each client uses the same pretrained feature extractor to generate class-specific embeddings (i.e., prototypes).
   - Different sampling strategies (mean, cluster, or random) can be chosen to capture representative embeddings of each class in each domain.

2. **Global Adapter Initialization**
   - The server collects these prototypes from all clients, effectively constructing a “prototype dataset.”
   - It then fine-tunes a global adapter on this aggregated dataset, simulating a centralized training approach without parameter averaging.
   - A single communication round suffices to train this global adapter.

3. **Few-shot Local Adaptation (Optional)**
   - If a client requires higher in-domain accuracy, it can use a small local dataset (few-shot) to further fine-tune the adapter.
   - Knowledge distillation (KD) is employed to maintain global knowledge while adapting to local data, mitigating catastrophic forgetting.



## Experiment

### Performance on multi-domain
- Compared with FedAvg, FedProx, MOON, Ditto, FedProto, and DBE on DomainNet and PACS, MPFT consistently achieves higher **in-domain** and **out-of-domain** accuracy.
- Notably, MPFT converges within **one communication round**, drastically reducing computational and communication overheads.

### Performance on each domain
- Per-domain analysis via radar charts shows MPFT maintains more “balanced” performance across distinct domains.
- It avoids large performance drops in certain domains and achieves good overall fairness in heterogeneous distributions.

### Impact of multi-domain differences on performance
- Even if each client contains mixed data from multiple domains, MPFT still outperforms other methods.
- As domain heterogeneity diminishes, the performance gap to baselines narrows, but MPFT remains strong.

### Performance with local adaptation
- When few-shot local data and KD are employed, clients can improve in-domain accuracy without severely sacrificing out-of-domain accuracy.
- Proper KD weighting strikes a balance between preserving global knowledge and optimizing local performance.



## Privacy Preservation Analysis
- MPFT applies differential privacy (via Gaussian noise) to client prototypes before uploading to the server, preventing adversaries from inferring raw data.
- Experiments with a feature space hijacking attack indicate that reconstructing the original images from shared prototypes is extremely difficult, even with knowledge of the pretrained encoder.
- Adding moderate noise can also mitigate overfitting and, in some cases, improve robustness without significant performance loss.



## Convergence Analysis
- The paper provides a theoretical analysis under non-convex conditions, showing that prototype-based fine-tuning converges in expectation.
- With appropriate learning rates and bounded prototype divergence, the loss decreases monotonically, ensuring convergence to a stationary point.



## Conclusion and Future Work
**Conclusion**
- MPFT addresses the shortcomings of naive parameter averaging in FDA by training on aggregated multi-domain prototypes.
- It achieves high out-of-domain performance while maintaining strong in-domain accuracy, requiring only a single round of communication for the global adapter.
- It is lightweight, privacy-preserving, and empirically robust to domain heterogeneity.

**Future Work**
1. **Prototype Quality**: Investigate improved prototype generation and better pretrained backbones, especially when within-class data variance is large.
2. **Advanced Privacy**: Explore stronger defenses against membership or attribute inference attacks while maintaining high performance.
3. **Real-World Extensions**: Adapt MPFT to more complex domains and tasks, such as financial fraud detection or clinical data analysis, where multi-domain data are prevalent.

![motivation](/static/image/motivation.pdf)
*Comparison of MPFT to centralized learning and previous averaging-based FL approaches.*

## Citation
```
@misc{zhang2024enhancingfederateddomainadaptation,
      title={Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning},
      author={Jingyuan Zhang and Yiyang Duan and Shuaicheng Niu and Yang Cao and Wei Yang Bryan Lim},
      year={2024},
      eprint={2410.07738},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.07738},
}
```

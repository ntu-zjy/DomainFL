---
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
---

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
The paper "On Computable Numbers, with an Application to the Entscheidungsproblem" was published by Alan Turing in 1936. In this groundbreaking paper, Turing introduced the concept of a universal computing machine, now known as the Turing machine.

## Objective
Turing's main objective in this paper was to investigate the notion of computability and its relation to the Entscheidungsproblem (the decision problem), which is concerned with determining whether a given mathematical statement is provable or not.

## Key Ideas
1. Turing first presented the concept of a "computable number," which refers to a number that can be computed by an algorithm or a definite step-by-step process.
2. He introduced the notion of a Turing machine, an abstract computational device consisting of an infinite tape divided into cells and a read-write head. The machine can read and write symbols on the tape, move the head left or right, and transition between states based on a set of rules.
3. Turing demonstrated that the set of computable numbers is enumerable, meaning it can be listed in a systematic way, even though it is not necessarily countable.
4. He proved the existence of non-computable numbers, which cannot be computed by any Turing machine.
5. Turing showed that the Entscheidungsproblem is undecidable, meaning there is no algorithm that can determine, for any given mathematical statement, whether it is provable or not.

![Turing Machine](/static/image/Turing_machine.png)

## Banner Image
![Banner Image](https://cdn.pixabay.com/photo/2018/06/24/08/17/colorful-abstract-background-3494093_1280.jpg)

::: caption
This is an image caption
:::

*Figure 1: A representation of a Turing Machine. Source: [Wiki](https://en.wikipedia.org/wiki/Turing_machine).*

## Table: Comparison of Computable and Non-Computable Numbers

| Computable Numbers | Non-Computable Numbers |
|-------------------|-----------------------|
| Rational numbers, e.g., 1/2, 3/4 | Transcendental numbers, e.g., π, e |
| Algebraic numbers, e.g., √2, ∛3 | Non-algebraic numbers, e.g., √2 + √3 |
| Numbers with finite decimal representations | Numbers with infinite, non-repeating decimal representations |

He used the concept of a universal Turing machine to prove that the set of computable functions is recursively enumerable, meaning it can be listed by an algorithm.

## Significance
Turing's paper laid the foundation for the theory of computation and had a profound impact on the development of computer science. The Turing machine became a fundamental concept in theoretical computer science, serving as a theoretical model for studying the limits and capabilities of computation. Turing's work also influenced the development of programming languages, algorithms, and the design of modern computers.

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

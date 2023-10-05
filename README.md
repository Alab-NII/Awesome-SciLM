# Awesome PLM for Scientific Text (SciLM)

## Content
- [Related survey papers](https://github.com/Alab-NII/Awesome-SciLM/tree/main#related-survey-papers)
- [Existing SciLMs](https://github.com/Alab-NII/Awesome-SciLM/tree/main#existing-scilms)
  - [Bio SciLMs](https://github.com/Alab-NII/Awesome-SciLM/tree/main#bio-scilms)
  - [Chemical SciLMs](https://github.com/Alab-NII/Awesome-SciLM/tree/main#chemical-scilms)
  - [Multi-domain SciLMs](https://github.com/Alab-NII/Awesome-SciLM/tree/main#multi-domain-scilms)
  - [Other domains SciLMs](https://github.com/Alab-NII/Awesome-SciLM/tree/main#other-domains-scilms)
- Awesome scientific datasets

## Related survey papers
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) - arXiv 2023
- [Large-scale Multi-Modal Pre-trained Models: A Comprehensive Survey](https://arxiv.org/abs/2302.10035) - Accepted by Machine Intelligence Research 2023
- [Pre-trained Language Models in Biomedical Domain: A Systematic Survey](https://arxiv.org/abs/2110.05006) - 	Accepted in ACM Computing Surveys 2023
- [AMMU: A survey of transformer-based biomedical pretrained language models](https://dl.acm.org/doi/abs/10.1016/j.jbi.2021.103982) - Journal of Biomedical Informatics 2022
- [Pre-Trained Language Models and Their Applications](https://www.sciencedirect.com/science/article/pii/S2095809922006324) - Engineering, 2022
- [Pre-trained models: Past, present and future](https://www.sciencedirect.com/science/article/pii/S2666651021000231) - AI Open, Volume 2, 2021


## Existing SciLMs

### Bio SciLMs
No. | Year | Name | Base-model | Objective | #Parameters | Code
|---|---| --- |---|---| --- |--- |
1 | 2019/01 | BioBERT | BERT | MLM, NSP | 110M | Github
2 | 2019/02 | BERT-MIMIC | BERT | MLM, NSP | 110M, 340M | Github
3 | 2019/04 | BioELMo | ELMo | Bi-LM | 93.6M | Github
4 | 2019/04 | Clinical BERT (Emily) | BERT | MLM, NSP | 110M | Github
5 | 2019/04 | ClinicalBERT (Kexin) | BERT | MLM, NSP | 110M | Github
6 | 2019/06 | BlueBERT | BERT | MLM, NSP | 110M, 340M | Github
7 | 2019/06 | G-BERT | GNN + BERT | Self-Prediction, Dual-Prediction | 3M | Github
8 | 2019/07 | BEHRT | BERT | MLM, NSP | N/A | Github
9 | 2019/08 | BioFLAIR | FLAIR | Bi-LM | N/A | Github
10 | 2019/09 | EhrBERT | BERT | MLM, NSP | 110M | Github
11 | 2019/12 | Clinical XLNet | XLNet | Generalized Autoregressive Pretraining | 110M | Github
12 | 2020/04 | GreenBioBERT | BERT | CBOW Word2Vec, Word Vector Space Alignment | 110M | Github
13 | 2020/05 | BERT-XML | BERT | MLM, NSP | N/A | Github
14 | 2020/05 | Bio-ELECTRA | ELECTRA | Replaced Token Prediction | 14M | Github
15 | 2020/05 | Med-BERT | BERT | MLM, Prolonged LOS Prediction | 110M | Github
16 | 2020/05 | ouBioBERT | BERT | MLM, NSP | 110M | Github
17 | 2020/07 | PubMedBERT | BERT | MLM, NSP, Whole-Word Masking | 110M | Github
18 | 2020/08 | MCBERT | BERT | MLM, NSP | 110M, 340M | Github
19 | 2020/09 | BioALBERT | ALBERT | MLM, SOP | 12M, 18M | Github
20 | 2020/09 | BRLTM | BERT | MLM | N/A | Github
21 | 2020/10 | BioMegatron | Megatron | MLM, NSP | 345M, 800M, 1.2B | Github
22 | 2020/10 | CharacterBERT | BERT + Character-CNN | MLM, NSP | 105M | Github
23 | 2020/10 | ClinicalTransformer | BERT - ALBERT - RoBERTa - ELECTRA | MLM, NSP - MLM, SOP - MLM - Replaced Token Prediction | 110M - 12M - 125M - 110M | Github
24 | 2020/10 | SapBERT | BERT | Multi-Similarity Loss | 110M | Github
25 | 2020/10 | UmlsBERT | BERT | MLM | 110M | Github
26 | 2020/11 | bert-for-radiology | BERT | MLM, NSP | 110M | Github
27 | 2020/11 | Bio-LM | RoBERTa | MLM | 125M, 355M | Github
28 | 2020/11 | CODER | PubMedBERT - mBERT | Contrastive Learning | 110M - 110M | Github
29 | 2020/11 | exBERT | BERT | MLM, NSP | N/A | Github
30 | 2020/12 | BioMedBERT | BERT | MLM, NSP | 340M | Github
31 | 2020/12 | LBERT | BERT | MLM, NSP | 110M | Github
32 | 2021/04 | CovidBERT | BioBERT | MLM, NSP | 110M | Github
33 | 2021/04 | ELECTRAMed | ELECTRA | Replaced Token Prediction | N/A | Github
34 | 2021/04 | KeBioLM | PubMedBERT | MLM, Entity Detection, Entity Linking | 110M | Github
35 | 2021/04 | SINA-BERT | BERT | MLM | 110M | Github
36 | 2021/05 | ProteinBERT | BERT | Corrupted Token, Annotation Prediction | 16M | Github
37 | 2021/05 | SciFive | T5 | Span Corruption Prediction | 220M, 770M | Github
38 | 2021/06 | BioELECTRA | ELECTRA | Replaced Token Prediction | 110M | Github
39 | 2021/06 | EntityBERT | BERT | Entity-centric MLM | 110M | Github
40 | 2021/07 | MedGPT | GPT-2 + GLU + RotaryEmbed | LM | N/A | Github
41 | 2021/08 | SMedBERT | SMedBERT | Masked Neighbor Modeling, Masked Mention Modeling, SOP, MLM | N/A | Github
42 | 2021/09 | Bio-cli | RoBERTa | MLM, Subword Masking or Whole Word Masking | 125M | Github
43 | 2021/11 | UTH-BERT | BERT | MLM, NSP | 110M | Github
44 | 2021/12 | ChestXRayBERT | BERT | MLM, NSP | 110M | Github
45 | 2021/12 | MedRoBERTa.nl | RoBERTa | MLM | 123M | Github
46 | 2021/12 | PubMedELECTRA | ELECTRA | Replaced Token Prediction | 110M, 335M | Github
47 | 2022/01 | Clinical-BigBird | BigBird | MLM | 166M | Github
48 | 2022/01 | Clinical-Longformer | Longformer | MLM | 149M | Github
49 | 2022/03 | BioLinkBERT | BERT | MLM, Document Relation Prediction | 110M, 340M | Github
50 | 2022/04 | BioBART | BART |  Text Infilling, Sentence Permutation  | 140M, 400M | Github
51 | 2022/05 | bsc-bio-ehr-es | RoBERTa | MLM | 125M | Github
52 | 2022/05 | PathologyBERT | BERT | MLM, NSP | 110M | Github
53 | 2022/06 | RadBERT | RoBERTa | MLM | 110M | Github
54 | 2022/06 | ViHealthBERT | BERT | MLM, NSP, Capitalized Prediction | 110M | Github
55 | 2022/07 | Clinical Flair | Flair | Character-level Bi-LM | N/A | Github
56 | 2022/08 | KM-BERT | BERT | MLM, NSP | 99M | Github
57 | 2022/09 | BioGPT | GPT | Autoregressive Language Model | 347M, 1.5B | Github
58 | 2022/10 | Bioberturk | BERT | MLM, NSP | N/A | Github
59 | 2022/10 | DRAGON | GreaseLM | MLM, KG Link Prediction | 360M | Github
60 | 2022/10 | UCSF-BERT | BERT | MLM, NSP | 135M | Github
61 | 2022/10 | ViPubmedT5 | ViT5 | Spans-masking learning | 220M | Github
62 | 2022/12 | ALIBERT | BERT | MLM | 110M | Github
63 | 2022/12 | BioMedLM | GPT2 | Autoregressive Language Model | 2.7B | Github
64 | 2022/12 | BioReader | T5 & RETRO | MLM | 229.5M | Github
65 | 2022/12 | clinicalT5 | T5 | Span-mask Denoising Objective | 220M, 770M | Github
66 | 2022/12 | Gatortron | BERT | MLM | 8.9B | Github
67 | 2022/12 | Med-PaLM | Flan-PaLM | Instruction Prompt Tuning | 540B | Github
68 | 2023/01 | clinical-T5 | T5 | Fill-in-the-blank-style denoising objective | 220M, 770M | Github
69 | 2023/01 | CPT-BigBird | BigBird | MLM | 166M | Github
70 | 2023/01 | CPT-Longformer | Longformer | MLM | 149M | Github
71 | 2023/02 | Bioformer | Bioformer | MLM, NSP | 43M | Github
72 | 2023/02 | Lightweight | DistilBERT | MLM, Knowledge Distillation | 65M, 25M, 18M, 15M | Github
73 | 2023/03 | RAMM | PubmedBERT | MLM, Contrastive Learning, Image-Text Matching | N/A | Github
74 | 2023/04 | DrBERT | RoBERTa | MLM | 110M | Github
75 | 2023/04 | MOTOR | BLIP | MLM, Contrastive Learning, Image-Text Matching | N/A | Github
76 | 2023/05 | BiomedGPT | BART backbone + BERT-encoder + GPT-decoder | MLM | 33M, 93M, 182M | Github
77 | 2023/05 | TurkRadBERT | BERT | MLM, NSP | 110M | Github
78 | 2023/06 | CamemBERT-bio | BERT | Whole Word MLM | 111M | Github
79 | 2023/06 | ClinicalGPT | T5 | Supervised Fine Tuning, Rank-based Training | N/A | Github
80 | 2023/06 | EriBERTa | RoBERTa | MLM | 125M | Github
81 | 2023/06 | PharmBERT | BERT | MLM | 110M | Github
82 | 2023/07 | BioNART | BERT | Non-AutoRegressive Model | 110M | Github
83 | 2023/07 | BIOptimus | BERT | MLM | 110M | Github
84 | 2023/07 | KEBLM | BERT | MLM, Contrastive Learning, Ranking Objective | N/A | Github


### Chemical SciLMs
No. | Year | Name | Base-model | Objective | #Parameters | Code
|---|---| --- |---|---| --- |--- |
1 | 2020/03 | [NukeBERT](https://arxiv.org/pdf/2003.13821.pdf) | BERT | MLM, NSP | 110M | [Github](https://github.com/ayushjain1144/NukeBERT)
2 | 2020/10 | ChemBERTa | RoBERTa | MLM | 125M | Github
3 | 2021/05 | NukeLM | SciBERT, RoBERTa | MLM | 125M, 355M, 110M | Github
4 | 2021/06 | ChemBERT | RoBERTa | MLM | 110M | Github
5 | 2021/09 | MatSciBERT | BERT | MLM | 110M | Github
6 | 2021/10 | MatBERT | BERT | MLM | 110M | Github
7 | 2022/05 | BatteryBERT | BERT, SciBERT | MLM | 110M | Github
8 | 2022/05 | ChemGPT | GPT | Autoregressive Language Model | 1B | Github
9 | 2022/08 | MaterialsBERT (Shetty) | PubMedBERT | MLM, NSP, Whole-Word Masking | 110M | Github
10 | 2022/08 | ProcessBERT | BERT | MLM, NSP | 110M | Github
11 | 2022/09 | ChemBERTa-2 | RoBERTa | MLM, Multi-task Regression | 125M | Github
12 | 2022/09 | MaterialBERT (Yoshitake) | BERT | MLM, NSP | 110M | Github
13 | 2023/08 | GIT-Mol | GIT-Former | Xmodal-Text Matching, Xmodal-Text Contrastive Learning | 700M | Github


### Multi-domain SciLMs
No. | Year | Name | Base-model | Objective | #Parameters | Code
|---|---| --- |---|---| --- |--- |
1 | 2019/11 | S2ORC-SciBERT | BERT | MLM, NSP | 110M | Github
2 | 2020/04 | SPECTER | BERT | Triple-loss | 110M | Github
3 | 2021/03 | OAG-BERT | BERT | MLM | 110M | Github
4 | 2022/05 | ScholarBERT | BERT | MLM | 770M | Github
5 | 2022/06 | SciDEBERTa | DeBERTa | MLM | N/A | Github
6 | 2022/09 | CSL-T5 | T5 | Fill-in-the-blank-style denoising objective | 220M | Github
7 | 2022/10 | AcademicRoBERTa | RoBERTa | MLM | 125M | Github
8 | 2022/11 | Galactica | GPT | Autoregressive Language Model | 125M, 1.3B, 6.7B, 30B, 120B | Github
9 | 2022/11 | VarMAE | RoBERTa | MLM | 110M | Github
10 | 2023/05 | Patton | GNN + BERT | Network-contextualized MLM, Masked Node Prediction | N/A | Github

### Other domains SciLMs
  

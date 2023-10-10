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
1 | 2019/01 | [BioBERT](https://arxiv.org/abs/1901.08746) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/dmis-lab/biobert) ![](https://img.shields.io/github/stars/dmis-lab/biobert?style=social)
2 | 2019/02 | [BERT-MIMIC](https://arxiv.org/abs/1902.08691) | BERT | MLM, NSP | 110M, 340M | N/A
3 | 2019/04 | [BioELMo](https://arxiv.org/abs/1904.02181) | ELMo | Bi-LM | 93.6M | [GitHub](https://github.com/Andy-jqa/bioelmo) ![](https://img.shields.io/github/stars/Andy-jqa/bioelmo?style=social)
4 | 2019/04 | [Clinical BERT (Emily)](https://arxiv.org/abs/1904.03323) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT) ![](https://img.shields.io/github/stars/EmilyAlsentzer/clinicalBERT?style=social)
5 | 2019/04 | [ClinicalBERT (Kexin)](https://arxiv.org/abs/1904.05342) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/kexinhuang12345/clinicalBERT) ![](https://img.shields.io/github/stars/kexinhuang12345/clinicalBERT?style=social)
6 | 2019/06 | [BlueBERT](https://arxiv.org/abs/1906.05474) | BERT | MLM, NSP | 110M, 340M | [GitHub](https://github.com/ncbi-nlp/bluebert) ![](https://img.shields.io/github/stars/ncbi-nlp/bluebert?style=social)
7 | 2019/06 | [G-BERT](https://arxiv.org/abs/1906.00346) | GNN + BERT | Self-Prediction, Dual-Prediction | 3M | [GitHub](https://github.com/jshang123/G-Bert) ![](https://img.shields.io/github/stars/jshang123/G-Bert?style=social)
8 | 2019/07 | [BEHRT](https://arxiv.org/abs/1907.09538) | BERT | MLM, NSP | N/A | [GitHub](https://github.com/deepmedicine/BEHRT) ![](https://img.shields.io/github/stars/deepmedicine/BEHRT?style=social)
9 | 2019/08 | [BioFLAIR](https://arxiv.org/abs/1908.05760) | FLAIR | Bi-LM | N/A | [GitHub](https://github.com/shreyashub/BioFLAIR) ![](https://img.shields.io/github/stars/shreyashub/BioFLAIR?style=social)
10 | 2019/09 | [EhrBERT](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6746103/) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/umassbento/ehrbert) ![](https://img.shields.io/github/stars/umassbento/ehrbert?style=social)
11 | 2019/12 | [Clinical XLNet](https://aclanthology.org/2020.clinicalnlp-1.11/) | XLNet | Generalized Autoregressive Pretraining | 110M | [GitHub](https://github.com/lindvalllab/clinicalXLNet) ![](https://img.shields.io/github/stars/lindvalllab/clinicalXLNet?style=social)
12 | 2020/04 | [GreenBioBERT](https://aclanthology.org/2020.findings-emnlp.134/) | BERT | CBOW Word2Vec, Word Vector Space Alignment | 110M | [GitHub](https://github.com/npoe/covid-qa/) ![](https://img.shields.io/github/stars/npoe/covid-qa?style=social)
13 | 2020/05 | [BERT-XML](https://aclanthology.org/2020.clinicalnlp-1.3/) | BERT | MLM, NSP | N/A | N/A
14 | 2020/05 | [Bio-ELECTRA](https://aclanthology.org/2020.sdp-1.12/) | ELECTRA | Replaced Token Prediction | 14M | [GitHub](https://github.com/SciCrunch/bio_electra) ![](https://img.shields.io/github/stars/SciCrunch/bio_electra?style=social)
15 | 2020/05 | [Med-BERT](https://www.nature.com/articles/s41746-021-00455-y) | BERT | MLM, Prolonged LOS Prediction | 110M | [GitHub](https://github.com/ZhiGroup/Med-BERT) ![](https://img.shields.io/github/stars/ZhiGroup/Med-BERT?style=social)
16 | 2020/05 | [ouBioBERT](https://arxiv.org/abs/2005.07202) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/sy-wada/blue_benchmark_with_transformers) ![](https://img.shields.io/github/stars/sy-wada/blue_benchmark_with_transformers?style=social)
17 | 2020/07 | [PubMedBERT](https://arxiv.org/abs/2007.15779) | BERT | MLM, NSP, Whole-Word Masking | 110M | [HuggingFace](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
18 | 2020/08 | [MCBERT](https://arxiv.org/abs/2008.10813) | BERT | MLM, NSP | 110M, 340M | [GitHub](https://github.com/alibaba-research/ChineseBLUE) ![](https://img.shields.io/github/stars/alibaba-research/ChineseBLUE?style=social)
19 | 2020/09 | [BioALBERT](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04688-w) | ALBERT | MLM, SOP | 12M, 18M | [GitHub](https://github.com/usmaann/BioALBERT) ![](https://img.shields.io/github/stars/usmaann/BioALBERT?style=social)
20 | 2020/09 | [BRLTM](https://arxiv.org/abs/2009.12656) | BERT | MLM | N/A | [GitHub](https://github.com/lanyexiaosa/brltm) ![](https://img.shields.io/github/stars/lanyexiaosa/brltm?style=social)
21 | 2020/10 | [BioMegatron](https://aclanthology.org/2020.emnlp-main.379/) | Megatron | MLM, NSP | 345M, 800M, 1.2B | [GitHub](https://github.com/NVIDIA/NeMo) ![](https://img.shields.io/github/stars/NVIDIA/NeMo?style=social)
22 | 2020/10 | [CharacterBERT](https://aclanthology.org/2020.coling-main.609/) | BERT + Character-CNN | MLM, NSP | 105M | [GitHub](https://github.com/helboukkouri/character-bert) ![](https://img.shields.io/github/stars/helboukkouri/character-bert?style=social)
23 | 2020/10 | [ClinicalTransformer](https://academic.oup.com/jamia/article-abstract/27/12/1935/5943218?redirectedFrom=fulltext) | BERT - ALBERT - RoBERTa - ELECTRA | MLM, NSP - MLM, SOP - MLM - Replaced Token Prediction | 110M - 12M - 125M - 110M | [GitHub](https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER) ![](https://img.shields.io/github/stars/uf-hobi-informatics-lab/ClinicalTransformerNER?style=social)
24 | 2020/10 | [SapBERT](https://aclanthology.org/2021.naacl-main.334/) | BERT | Multi-Similarity Loss | 110M | [GitHub](https://github.com/cambridgeltl/sapbert) ![](https://img.shields.io/github/stars/cambridgeltl/sapbert?style=social)
25 | 2020/10 | [UmlsBERT](https://aclanthology.org/2021.naacl-main.139/) | BERT | MLM | 110M | [GitHub](https://github.com/gmichalo/UmlsBERT) ![](https://img.shields.io/github/stars/gmichalo/UmlsBERT?style=social)
26 | 2020/11 | [bert-for-radiology](https://academic.oup.com/bioinformatics/article/36/21/5255/5875602) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/rAIdiance/bert-for-radiology) ![](https://img.shields.io/github/stars/rAIdiance/bert-for-radiology?style=social)
27 | 2020/11 | [Bio-LM](https://aclanthology.org/2020.clinicalnlp-1.17/) | RoBERTa | MLM | 125M, 355M | [GitHub](https://github.com/facebookresearch/bio-lm) ![](https://img.shields.io/github/stars/facebookresearch/bio-lm?style=social)
28 | 2020/11 | [CODER](https://arxiv.org/abs/2011.02947) | PubMedBERT - mBERT | Contrastive Learning | 110M - 110M | [GitHub](https://github.com/GanjinZero/CODER) ![](https://img.shields.io/github/stars/GanjinZero/CODER?style=social)
29 | 2020/11 | [exBERT](https://aclanthology.org/2020.findings-emnlp.129/) | BERT | MLM, NSP | N/A | [GitHub](https://github.com/cgmhaicenter/exBERT) ![](https://img.shields.io/github/stars/cgmhaicenter/exBERT?style=social)
30 | 2020/12 | [BioMedBERT](https://aclanthology.org/2020.coling-main.59/) | BERT | MLM, NSP | 340M | [GitHub](https://github.com/BioMedBERT/biomedbert) ![](https://img.shields.io/github/stars/BioMedBERT/biomedbert?style=social)
31 | 2020/12 | [LBERT](https://pubmed.ncbi.nlm.nih.gov/33331647/) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/warikoone/LBERT) ![](https://img.shields.io/github/stars/warikoone/LBERT?style=social)
32 | 2021/04 | [CovidBERT](https://journals.flvc.org/FLAIRS/article/download/128488/130074) | BioBERT | MLM, NSP | 110M | N/A
33 | 2021/04 | [ELECTRAMed](https://arxiv.org/abs/2104.09585) | ELECTRA | Replaced Token Prediction | N/A | [GitHub](https://github.com/gmpoli/electramed) ![](https://img.shields.io/github/stars/gmpoli/electramed?style=social)
34 | 2021/04 | [KeBioLM](https://aclanthology.org/2021.bionlp-1.20/) | PubMedBERT | MLM, Entity Detection, Entity Linking | 110M | [GitHub](https://github.com/GanjinZero/KeBioLM) ![](https://img.shields.io/github/stars/GanjinZero/KeBioLM?style=social)
35 | 2021/04 | [SINA-BERT](https://arxiv.org/abs/2104.07613) | BERT | MLM | 110M | N/A
36 | 2021/05 | [ProteinBERT](https://academic.oup.com/bioinformatics/article/38/8/2102/6502274) | BERT | Corrupted Token, Annotation Prediction | 16M | [GitHub](https://github.com/nadavbra/protein_bert) ![](https://img.shields.io/github/stars/nadavbra/protein_bert?style=social)
37 | 2021/05 | [SciFive](https://arxiv.org/abs/2106.03598) | T5 | Span Corruption Prediction | 220M, 770M | [GitHub](https://github.com/justinphan3110/SciFive) ![](https://img.shields.io/github/stars/justinphan3110/SciFive?style=social)
38 | 2021/06 | [BioELECTRA](https://aclanthology.org/2021.bionlp-1.16/) | ELECTRA | Replaced Token Prediction | 110M | [GitHub](https://github.com/kamalkraj/BioELECTRA) ![](https://img.shields.io/github/stars/kamalkraj/BioELECTRA?style=social)
39 | 2021/06 | [EntityBERT](https://aclanthology.org/2021.bionlp-1.21/) | BERT | Entity-centric MLM | 110M | N/A
40 | 2021/07 | [MedGPT](https://arxiv.org/abs/2107.03134) | GPT-2 + GLU + RotaryEmbed | LM | N/A | N/A
41 | 2021/08 | [SMedBERT](https://arxiv.org/abs/2108.08983) | SMedBERT | Masked Neighbor Modeling, Masked Mention Modeling, SOP, MLM | N/A | [GitHub](https://github.com/MatNLP/SMedBERT) ![](https://img.shields.io/github/stars/MatNLP/SMedBERT?style=social)
42 | 2021/09 | [Bio-cli](https://arxiv.org/abs/2109.03570) | RoBERTa | MLM, Subword Masking or Whole Word Masking | 125M | [GitHub](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es) ![](https://img.shields.io/github/stars/PlanTL-GOB-ES/lm-biomedical-clinical-es?style=social)
43 | 2021/11 | [UTH-BERT](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259763) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/jinseikenai/uth-bert) ![](https://img.shields.io/github/stars/jinseikenai/uth-bert?style=social)
44 | 2021/12 | [ChestXRayBERT](https://ieeexplore.ieee.org/abstract/document/9638337) | BERT | MLM, NSP | 110M | N/A
45 | 2021/12 | [MedRoBERTa.nl](https://clinjournal.org/clinj/article/view/132) | RoBERTa | MLM | 123M | [GitHub](https://github.com/cltl-students/verkijk_stella_rma_thesis_dutch_medical_language_model) ![](https://img.shields.io/github/stars/cltl-students/verkijk_stella_rma_thesis_dutch_medical_language_model?style=social)
46 | 2021/12 | [PubMedELECTRA](https://arxiv.org/abs/2112.07869) | ELECTRA | Replaced Token Prediction | 110M, 335M | [HuggingFace](https://huggingface.co/microsoft/BiomedNLP-PubMedELECTRA-base-uncased-abstract)
47 | 2022/01 | [Clinical-BigBird](https://arxiv.org/abs/2201.11838) | BigBird | MLM | 166M | [GitHub](https://github.com/luoyuanlab/Clinical-Longformer) ![](https://img.shields.io/github/stars/luoyuanlab/Clinical-Longformer?style=social)
48 | 2022/01 | [Clinical-Longformer](https://arxiv.org/abs/2201.11838) | Longformer | MLM | 149M | [GitHub](https://github.com/luoyuanlab/Clinical-Longformer) ![](https://img.shields.io/github/stars/luoyuanlab/Clinical-Longformer?style=social)
49 | 2022/03 | [BioLinkBERT](https://aclanthology.org/2022.acl-long.551/) | BERT | MLM, Document Relation Prediction | 110M, 340M | [GitHub](https://github.com/michiyasunaga/LinkBERT) ![](https://img.shields.io/github/stars/michiyasunaga/LinkBERT?style=social)
50 | 2022/04 | [BioBART](https://aclanthology.org/2022.bionlp-1.9/) | BART |  Text Infilling, Sentence Permutation  | 140M, 400M | [GitHub](https://github.com/GanjinZero/BioBART) ![](https://img.shields.io/github/stars/GanjinZero/BioBART?style=social)
51 | 2022/05 | [bsc-bio-ehr-es](https://aclanthology.org/2022.bionlp-1.19/) | RoBERTa | MLM | 125M | [GitHub](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es) ![](https://img.shields.io/github/stars/PlanTL-GOB-ES/lm-biomedical-clinical-es?style=social)
52 | 2022/05 | [PathologyBERT](https://arxiv.org/abs/2205.06885) | BERT | MLM, NSP | 110M | [HuggingFace](https://huggingface.co/tsantos/PathologyBERT)
53 | 2022/06 | [RadBERT](https://pubs.rsna.org/doi/10.1148/ryai.210258) | RoBERTa | MLM | 110M | [GitHub](https://github.com/zzxslp/RadBERT) ![](https://img.shields.io/github/stars/zzxslp/RadBERT?style=social)
54 | 2022/06 | [ViHealthBERT](https://aclanthology.org/2022.lrec-1.35/) | BERT | MLM, NSP, Capitalized Prediction | 110M | [GitHub](https://github.com/demdecuong/vihealthbert) ![](https://img.shields.io/github/stars/demdecuong/vihealthbert?style=social)
55 | 2022/07 | [Clinical Flair](https://aclanthology.org/2022.clinicalnlp-1.9/) | Flair | Character-level Bi-LM | N/A | [GitHub](https://github.com/plncmm/spanish-clinical-flair) ![](https://img.shields.io/github/stars/plncmm/spanish-clinical-flair?style=social)
56 | 2022/08 | [KM-BERT](https://link.springer.com/content/pdf/10.1038/s41598-022-17806-8.pdf) | BERT | MLM, NSP | 99M | [GitHub](https://github.com/KU-RIAS/KM-BERT-Korean-Medical-BERT) ![](https://img.shields.io/github/stars/KU-RIAS/KM-BERT-Korean-Medical-BERT?style=social)
57 | 2022/09 | [BioGPT](https://arxiv.org/abs/2210.10341) | GPT | Autoregressive Language Model | 347M, 1.5B | [GitHub](https://github.com/microsoft/BioGPT) ![](https://img.shields.io/github/stars/microsoft/BioGPT?style=social)
58 | 2022/10 | [Bioberturk](https://www.researchsquare.com/article/rs-2165226/v1) | BERT | MLM, NSP | N/A | [GitHub](https://github.com/hazalturkmen/BioBERTurk) ![](https://img.shields.io/github/stars/hazalturkmen/BioBERTurk?style=social)
59 | 2022/10 | [DRAGON](https://arxiv.org/abs/2210.09338) | GreaseLM | MLM, KG Link Prediction | 360M | [GitHub](https://github.com/michiyasunaga/dragon) ![](https://img.shields.io/github/stars/michiyasunaga/dragon?style=social)
60 | 2022/10 | [UCSF-BERT](https://arxiv.org/abs/2210.06566) | BERT | MLM, NSP | 135M | N/A
61 | 2022/10 | [ViPubmedT5](https://arxiv.org/abs/2210.05598) | ViT5 | Spans-masking learning | 220M | [GitHub](https://github.com/vietai/ViPubmed) ![](https://img.shields.io/github/stars/vietai/ViPubmed?style=social)
62 | 2022/12 | [ALIBERT](https://hal.science/hal-03911564/) | BERT | MLM | 110M | N/A
63 | 2022/12 | [BioMedLM](https://crfm.stanford.edu/2022/12/15/biomedlm.html) | GPT2 | Autoregressive Language Model | 2.7B | [GitHub](https://github.com/stanford-crfm/BioMedLM) ![](https://img.shields.io/github/stars/stanford-crfm/BioMedLM?style=social)
64 | 2022/12 | [BioReader](https://aclanthology.org/2022.emnlp-main.390/) | T5 & RETRO | MLM | 229.5M | [GitHub](https://github.com/disi-unibo-nlp/bio-reader) ![](https://img.shields.io/github/stars/disi-unibo-nlp/bio-reader?style=social)
65 | 2022/12 | [clinicalT5](https://aclanthology.org/2022.findings-emnlp.398/) | T5 | Span-mask Denoising Objective | 220M, 770M | N/A
66 | 2022/12 | [Gatortron](https://www.nature.com/articles/s41746-022-00742-2) | BERT | MLM | 8.9B | [GitHub](https://github.com/uf-hobi-informatics-lab/GatorTron) ![](https://img.shields.io/github/stars/uf-hobi-informatics-lab/GatorTron?style=social)
67 | 2022/12 | [Med-PaLM](https://arxiv.org/abs/2212.13138) | Flan-PaLM | Instruction Prompt Tuning | 540B | [Official Site](https://sites.research.google/med-palm/)
68 | 2023/01 | [clinical-T5](https://www.physionet.org/content/clinical-t5/1.0.0/) | T5 | Fill-in-the-blank-style denoising objective | 220M, 770M | [PhysioNet](https://www.physionet.org/content/clinical-t5/1.0.0/)
69 | 2023/01 | [CPT-BigBird](https://scholarspace.manoa.hawaii.edu/items/3ce4b211-deda-46a3-a1df-da4827ca3c80) | BigBird | MLM | 166M | N/A
70 | 2023/01 | [CPT-Longformer](https://scholarspace.manoa.hawaii.edu/items/3ce4b211-deda-46a3-a1df-da4827ca3c80) | Longformer | MLM | 149M | N/A
71 | 2023/02 | [Bioformer](https://arxiv.org/abs/2302.01588) | Bioformer | MLM, NSP | 43M | [GitHub](https://github.com/WGLab/bioformer) ![](https://img.shields.io/github/stars/WGLab/bioformer?style=social)
72 | 2023/02 | [Lightweight](https://arxiv.org/abs/2302.04725) | DistilBERT | MLM, Knowledge Distillation | 65M, 25M, 18M, 15M | [GitHub](https://github.com/nlpie-research/Lightweight-Clinical-Transformers) ![](https://img.shields.io/github/stars/nlpie-research/Lightweight-Clinical-Transformers?style=social)
73 | 2023/03 | [RAMM](https://arxiv.org/abs/2303.00534) | PubmedBERT | MLM, Contrastive Learning, Image-Text Matching | N/A | [GitHub](https://github.com/GanjinZero/RAMM) ![](https://img.shields.io/github/stars/GanjinZero/RAMM?style=social)
74 | 2023/04 | [DrBERT](https://arxiv.org/abs/2304.00958) | RoBERTa | MLM | 110M | [GitHub](https://github.com/qanastek/DrBERT) ![](https://img.shields.io/github/stars/qanastek/DrBERT?style=social)
75 | 2023/04 | [MOTOR](https://arxiv.org/abs/2304.14204) | BLIP | MLM, Contrastive Learning, Image-Text Matching | N/A | [GitHub](https://github.com/chenzcv7/MOTOR) ![](https://img.shields.io/github/stars/chenzcv7/MOTOR?style=social)
76 | 2023/05 | [BiomedGPT](https://arxiv.org/abs/2305.17100) | BART backbone + BERT-encoder + GPT-decoder | MLM | 33M, 93M, 182M | [GitHub](https://github.com/taokz/BiomedGPT) ![](https://img.shields.io/github/stars/taokz/BiomedGPT?style=social)
77 | 2023/05 | [TurkRadBERT](https://arxiv.org/abs/2305.03788) | BERT | MLM, NSP | 110M | N/A
78 | 2023/06 | [CamemBERT-bio](https://arxiv.org/abs/2306.15550) | BERT | Whole Word MLM | 111M | [HuggingFace](https://huggingface.co/almanach/camembert-bio-base)
79 | 2023/06 | [ClinicalGPT](https://arxiv.org/abs/2306.09968) | T5 | Supervised Fine Tuning, Rank-based Training | N/A | N/A
80 | 2023/06 | [EriBERTa](https://arxiv.org/abs/2306.07373) | RoBERTa | MLM | 125M | N/A
81 | 2023/06 | [PharmBERT](https://academic.oup.com/bib/article-abstract/24/4/bbad226/7197744) | BERT | MLM | 110M | [GitHub](https://github.com/TahaAslani/PharmBERT) ![](https://img.shields.io/github/stars/TahaAslani/PharmBERT?style=social)
82 | 2023/07 | [BioNART](https://aclanthology.org/2023.bionlp-1.34/) | BERT | Non-AutoRegressive Model | 110M | [GitHub](https://github.com/aistairc/BioNART) ![](https://img.shields.io/github/stars/aistairc/BioNART?style=social)
83 | 2023/07 | [BIOptimus](https://aclanthology.org/2023.bionlp-1.31/) | BERT | MLM | 110M | [GitHub](https://github.com/rttl-ai/bioptimus) ![](https://img.shields.io/github/stars/rttl-ai/bioptimus?style=social)
84 | 2023/07 | [KEBLM](https://blender.cs.illinois.edu/paper/biomedicallm2023.pdf) | BERT | MLM, Contrastive Learning, Ranking Objective | N/A | N/A


### Chemical SciLMs
No. | Year | Name | Base-model | Objective | #Parameters | Code
|---|---| --- |---|---| --- |--- |
1 | 2020/03 | [NukeBERT](https://arxiv.org/abs/2003.13821) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/ayushjain1144/NukeBERT) ![](https://img.shields.io/github/stars/ayushjain1144/NukeBERT?style=social)
2 | 2020/10 | [ChemBERTa](https://arxiv.org/abs/2010.09885) | RoBERTa | MLM | 125M | [GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry) ![](https://img.shields.io/github/stars/seyonechithrananda/bert-loves-chemistry?style=social)
3 | 2021/05 | [NukeLM](https://arxiv.org/abs/2105.12192) | SciBERT, RoBERTa | MLM | 125M, 355M, 110M | [GitHub](https://github.com/pnnl/NUKELM) ![](https://img.shields.io/github/stars/pnnl/NUKELM?style=social)
4 | 2021/06 | [ChemBERT](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00284) | RoBERTa | MLM | 110M | [GitHub](https://github.com/jiangfeng1124/ChemRxnExtractor) ![](https://img.shields.io/github/stars/jiangfeng1124/ChemRxnExtractor?style=social)
5 | 2021/09 | [MatSciBERT](https://www.nature.com/articles/s41524-022-00784-w) | BERT | MLM | 110M | [GitHub](https://github.com/M3RG-IITD/MatSciBERT) ![](https://img.shields.io/github/stars/M3RG-IITD/MatSciBERT?style=social)
6 | 2021/10 | [MatBERT](https://www.cell.com/patterns/pdf/S2666-3899(22)00073-3.pdf) | BERT | MLM | 110M | [GitHub](https://github.com/lbnlp/MatBERT) ![](https://img.shields.io/github/stars/lbnlp/MatBERT?style=social)
7 | 2022/05 | [BatteryBERT](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00035) | BERT, SciBERT | MLM | 110M | [GitHub](https://github.com/ShuHuang/batterybert) ![](https://img.shields.io/github/stars/ShuHuang/batterybert?style=social)
8 | 2022/05 | [ChemGPT](https://chemrxiv.org/engage/chemrxiv/article-details/627bddd544bdd532395fb4b5) | GPT | Autoregressive Language Model | 1B | [GitHub](https://github.com/ncfrey/litmatter) ![](https://img.shields.io/github/stars/ncfrey/litmatter?style=social)
9 | 2022/08 | [MaterialsBERT (Shetty)](https://arxiv.org/abs/2209.13136) | PubMedBERT | MLM, NSP, Whole-Word Masking | 110M | [GitHub](https://github.com/Ramprasad-Group/polymer_information_extraction) ![](https://img.shields.io/github/stars/Ramprasad-Group/polymer_information_extraction?style=social)
10 | 2022/08 | [ProcessBERT](https://www.sciencedirect.com/science/article/pii/S2405896322009740) | BERT | MLM, NSP | 110M | N/A
11 | 2022/09 | [ChemBERTa-2](https://arxiv.org/abs/2209.01712) | RoBERTa | MLM, Multi-task Regression | 125M | [GitHub](https://github.com/seyonechithrananda/bert-loves-chemistry) ![](https://img.shields.io/github/stars/seyonechithrananda/bert-loves-chemistry?style=social)
12 | 2022/09 | [MaterialBERT (Yoshitake)](https://mdr.nims.go.jp/concern/publications/pc289n449?locale=en) | BERT | MLM, NSP | 110M | [MDR](https://mdr.nims.go.jp/concern/publications/pc289n449?locale=en)
13 | 2023/08 | [GIT-Mol](https://arxiv.org/abs/2308.06911) | GIT-Former | Xmodal-Text Matching, Xmodal-Text Contrastive Learning | 700M | N/A


### Multi-domain SciLMs
No. | Year | Name | Base-model | Objective | #Parameters | Code
|---|---| --- |---|---| --- |--- |
|1|2019/03| [SciBERT](https://aclanthology.org/D19-1371/) (CS + Bio)|BERT|MLM, NSP| 110M | [GitHub](https://github.com/allenai/scibert) ![](https://img.shields.io/github/stars/allenai/scibert?style=social)
2 | 2019/11 | [S2ORC-SciBERT](https://aclanthology.org/2020.acl-main.447/) | BERT | MLM, NSP | 110M | [GitHub](https://github.com/allenai/s2orc) ![](https://img.shields.io/github/stars/allenai/s2orc?style=social)
3 | 2020/04 | [SPECTER](https://aclanthology.org/2020.acl-main.207/) | BERT | Triple-loss | 110M | [GitHub](https://github.com/allenai/specter) ![](https://img.shields.io/github/stars/allenai/specter?style=social)
4 | 2021/03 | [OAG-BERT](https://arxiv.org/abs/2103.02410) | BERT | MLM | 110M | [GitHub](https://github.com/THUDM/OAG-BERT) ![](https://img.shields.io/github/stars/THUDM/OAG-BERT?style=social)
5 | 2022/05 | [ScholarBERT](https://arxiv.org/abs/2205.11342v1) | BERT | MLM | 770M | [HuggingFace](https://huggingface.co/globuslabs/ScholarBERT)
6 | 2022/06 | [SciDEBERTa](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9791256) | DeBERTa | MLM | N/A | [GitHub](https://github.com/Eunhui-Kim/SciDeBERTa-Fine-Tuning) ![](https://img.shields.io/github/stars/Eunhui-Kim/SciDeBERTa-Fine-Tuning?style=social)
7 | 2022/09 | [CSL-T5](https://arxiv.org/abs/2209.05034) | T5 | Fill-in-the-blank-style denoising objective | 220M | [GitHub](https://github.com/ydli-ai/CSL) ![](https://img.shields.io/github/stars/ydli-ai/CSL?style=social)
8 | 2022/10 | [AcademicRoBERTa](https://aclanthology.org/2022.sdp-1.16/) | RoBERTa | MLM | 125M | [GitHub](https://github.com/EhimeNLP/AcademicRoBERTa) ![](https://img.shields.io/github/stars/EhimeNLP/AcademicRoBERTa?style=social)
9 | 2022/11 | [Galactica](https://arxiv.org/abs/2211.09085) | GPT | Autoregressive Language Model | 125M, 1.3B, 6.7B, 30B, 120B | [GitHub](https://github.com/paperswithcode/galai) ![](https://img.shields.io/github/stars/paperswithcode/galai?style=social)
10 | 2022/11 | [VarMAE](https://arxiv.org/abs/2211.00430) | RoBERTa | MLM | 110M | N/A
11 | 2023/05 | [Patton](https://arxiv.org/abs/2305.12268) | GNN + BERT | Network-contextualized MLM, Masked Node Prediction | N/A | GitHub

### Other domains SciLMs
Sorted by Domain-name

No. | Year | Name | Base-model | Objective | #Parameters | Code | Domain
|---|---| --- |---|---| --- |--- |--- |
2022/04  |  SecureBERT  |  RoBERTa  |  MLM  |  CP  |  125M  |  GitHub| Cybersecurity
2022/12  |  CySecBERT  |  BERT  |  MLM, NSP  |  CP  |  110M  | GitHub| Cybersecurity
2021/05  |  MathBERT (Peng)  |  BERT  |  MLM, Masked Substructure Prediction, Context Correspondence Prediction  |  CP  |  110M  | GitHub| Math
2021/06  |  MathBERT (Shen)  |  RoBERTa  |  MLM  |  CP  |  110M  | GitHub| Math
2021/10  |  ClimateBert  |  DistilROBERTA  |  MLM  |  CP  |  66M  | GitHub| Climate
2020/02  |  SciGPT2  |  GPT2  |  LM  |  CP  |  124M  | GitHub| CS
2023/06  |  K2  |  LLaMA  |  Cosine Loss  |  CP  |  7B  | GitHub| Geoscience
2023/03  |  ManuBERT  |  BERT  |  MLM  |  CP  |  110M, 126M  | GitHub| Manufaturing
2023/01  |  ProtST  |  BERT  |  Masked Protein Modeling, Contrastive Learning, Multi-modal Masked Prediction  |  CP & FS  |  N/A  | GitHub| Protein
2023/01  |  SciEdBERT  |  BERT  |  MLM  |  CP  |  110M  | GitHub| Science Education
2022/06  |  SsciBERT  |  BERT  |  MLM, NSP  |  CP  |  110M  | GitHub| Social Science

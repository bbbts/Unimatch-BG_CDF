
# UniMatch + BG-CDF

**Integrating Boundary-Guided Class Distribution Flow (BG-CDF) into UniMatch for Semi-Supervised Semantic Segmentation**

This repository extends the original [UniMatch](https://github.com/LiheYoung/UniMatch) framework with **Boundary-Guided Class Distribution Flow (BG-CDF)**, a boundary-aware regularization method designed to improve semantic segmentation near class boundaries.

The implementation is evaluated on **Cityscapes** using DeepLabV3+ with a ResNet-101 backbone. We compare a reproduced UniMatch baseline against UniMatch augmented with BG-CDF under four labeled-data settings.

- **Original method:** UniMatch
- **Extended method:** UniMatch + BG-CDF
- **Dataset:** Cityscapes
- **Backbone:** ResNet-101
- **BG-CDF loss weight:** λ_BG-CDF = 0.5
- **Evaluation metric:** Validation mean Intersection over Union (mIoU)

## Repository

- **Modified implementation:** [Unimatch-BG_CDF](https://github.com/bbbts/Unimatch-BG_CDF)
- **Training logs and model checkpoints:** [Google Drive — Cityscapes experiments](https://drive.google.com/drive/u/1/folders/1Jveuhi3S4HCRj3FG1ov__xOW4rMQRIey)
- **Original UniMatch implementation:** [LiheYoung/UniMatch](https://github.com/LiheYoung/UniMatch)

## Method Overview

UniMatch is a semi-supervised semantic segmentation framework that learns from labeled and unlabeled images using weak-to-strong consistency.

This repository extends the training objective with BG-CDF, a boundary-aware loss that encourages more consistent class-probability distributions around predicted semantic boundaries.

The combined objective can be expressed conceptually as:

L_total = L_UniMatch + λ_BG-CDF × L_BG-CDF

where L_UniMatch denotes the original UniMatch training objective and L_BG-CDF is the additional boundary-aware regularization term.

For the experiments reported here, the BG-CDF loss weight is set to **0.5**.

The goal is to investigate whether explicitly incorporating boundary information can improve segmentation performance when only a fraction of the training images are labeled.

## Cityscapes Experiments

### Experimental Setup

| Component | Configuration |
|---|---|
| Dataset | Cityscapes |
| Segmentation architecture | DeepLabV3+ |
| Backbone | ResNet-101 |
| Training framework | UniMatch |
| Proposed extension | UniMatch + BG-CDF |
| BG-CDF loss weight | 0.5 |
| Labeled-data settings | 1/16, 1/8, 1/4, 1/2 |
| Training duration | 240 epochs |
| GPUs used | 2 |
| Main evaluation metric | Validation mIoU |

The original UniMatch implementation was independently reproduced in a reduced GPU configuration. The training setup used two GPUs rather than the four GPUs recommended for reproducing the original benchmark results. Consequently, these results should be interpreted in the context of the specific reproduction and training configuration used here.

### Results

The following table reports the best validation mIoU recorded during training for each labeled-data setting.

| Labeled Data | Reproduced UniMatch | UniMatch + BG-CDF | Absolute Change |
|:---:|---:|---:|---:|
| 1/16 | 75.57 | **75.73** | +0.16 |
| 1/8 | 77.90 | **78.18** | +0.28 |
| 1/4 | 78.88 | **79.17** | +0.29 |
| 1/2 | 79.50 | **79.55** | +0.05 |

*Scores are in percentage points of mIoU. Absolute change is calculated as the BG-CDF result minus the reproduced UniMatch result.*

### Observations

- BG-CDF improves the best recorded validation mIoU across all four labeled-data settings in these experiments.
- The largest absolute improvement is observed with the 1/4 labeled-data setting.
- The improvement at the 1/2 setting is comparatively small, suggesting that the effect of boundary-aware regularization may depend on the amount of labeled data available.
- These results provide preliminary evidence that BG-CDF can complement UniMatch. Additional runs with multiple random seeds and consistent data splits are needed to assess the robustness and statistical significance of the improvements.

**Important evaluation note:** The reported values are the best validation mIoU values recorded by the training process. They are not results from a separate, independently implemented inference or test-set evaluation. Checkpoints and training logs are available in the linked Google Drive folder.

## Training

### Requirements

Please refer to the original [UniMatch repository](https://github.com/LiheYoung/UniMatch) and the `requirements.txt` file in this repository for the environment and dependencies.

You will also need the Cityscapes dataset, the appropriate pretrained ResNet-101 weights, and a valid dataset path in the configuration.

### Train UniMatch + BG-CDF

The command used for the reported experiments was:

    CUDA_VISIBLE_DEVICES=2,3 sh scripts/train.sh 2 29502

Here:

- `CUDA_VISIBLE_DEVICES=2,3` selects the two visible GPUs.
- `2` specifies the number of GPUs passed to the training script.
- `29502` specifies the distributed training port.

The training script uses the repository's configured dataset, labeled-data split, model architecture, and training settings. Verify the relevant configuration files before running experiments with different settings.

### BG-CDF Configuration

The BG-CDF loss contribution was configured with a weight of `0.5`.

To reproduce the reported extension, verify the BG-CDF implementation and its configured weight in the training code. For a fair comparison, keep the dataset split, backbone, training schedule, and other relevant settings consistent between UniMatch and UniMatch + BG-CDF.

### Evaluation and Checkpoints

The training process evaluates the model during training and records validation mIoU. The best validation result is used for the reported comparison.

The original repository does not provide a separate inference script specifically for this experiment. Training logs and the corresponding model checkpoints are provided in the [Google Drive experiment folder](https://drive.google.com/drive/u/1/folders/1Jveuhi3S4HCRj3FG1ov__xOW4rMQRIey).

## Reproducibility Notes

- The results in this README come from the experiments conducted with this modified repository.
- The reproduced UniMatch baseline is distinguished from the original authors' published benchmark results.
- The experiments use a two-GPU configuration, so they should not be assumed to reproduce the original four-GPU setup exactly.
- The BG-CDF coefficient is 0.5 for the reported extension.
- The reported metric is the best validation mIoU recorded during training.
- Multiple random seeds, additional ablation studies, and independent evaluation would help establish the robustness and generality of the observed improvements.

## Original UniMatch Work

This project builds upon the following publication:

**Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation**

Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, and Yinghuan Shi. CVPR 2023.

Paper: [https://arxiv.org/abs/2208.09910](https://arxiv.org/abs/2208.09910)

Original implementation: [https://github.com/LiheYoung/UniMatch](https://github.com/LiheYoung/UniMatch)

### Citation

If you use the original UniMatch framework, please cite:

    @inproceedings{unimatch,
      title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
      author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
      booktitle={CVPR},
      year={2023}
    }

For work using the BG-CDF extension, please also cite the relevant BG-CDF publication when it becomes available.

## Acknowledgments

This repository builds on the publicly available [UniMatch](https://github.com/LiheYoung/UniMatch) implementation. We thank the original authors for making their code and research available.

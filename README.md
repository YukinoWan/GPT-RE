# GPT-RE: In-context Learning for Relation Extraction using Large Language Models

This is the official repository for the paper **"GPT-RE: In-context Learning for Relation Extraction using Large Language Models"** (EMNLP 2023).

## Updates

**[New]** The **GPT-RE with Fine-Tuned representations (GPT-RE_FT)** method is now available! The code has been reproduced and organized in our follow-up work. Please visit:

> **[GPT-RE-FT Repository](https://github.com/HanPT831/GPT-RE-FT)**

This repository contains the complete implementation including fine-tuned representation methods that were not included in the original release.

## Supported Features

This repository supports the following methods from the paper:
- **GPT-Random**: Random demonstration selection
- **GPT-SimCSE**: SimCSE-based similar demonstration retrieval
- **Reasoning Logic**: Task-aware reasoning in demonstrations
- **Entity-aware Similarity**: Entity information enhanced retrieval

## Usage

```bash
bash run_relation_ace.sh
```

## Configuration Options

In the example script file `run_relation_ace.sh`, the following options are available:

### Basic Settings
| Option | Description | Example |
|--------|-------------|---------|
| `--task` | Name of the task | `semeval`, `ace05`, `scierc` |
| `--model` | GPT model name | `text-davinci-003` |
| `--seed` | Random seed | `42` |

### Data Settings
| Option | Description |
|--------|-------------|
| `--num_test` | Number of test examples (should be smaller than test dataset size) |
| `--example_dataset` | File path for demonstration examples |
| `--test_dataset` | Path to test data (use `test.json` for full test set) |

### Demonstration Settings
| Option | Description |
|--------|-------------|
| `--fixed_example` | `1`: Fixed demonstrations; `0`: Re-retrieve for each test (use `0` for kNN) |
| `--fixed_test` | `1`: Fixed test dataset (can be ignored, use `--test_dataset` instead) |
| `--num_per_rel` | Examples per relation type for demonstrations (use `0` for kNN) |
| `--num_na` | NA examples for demonstrations (use `0` for w/o NA and kNN setups) |
| `--num_run` | Keep `1` |

### Method Settings
| Option | Description |
|--------|-------------|
| `--random_label` | `1`: Use random labels in demonstrations |
| `--reasoning` | `1`: Add reasoning to demonstrations |
| `--use_knn` | `1`: Use kNN for demonstration retrieval |
| `--k` | Top-k for kNN retrieval |
| `--reverse` | `1`: Reverse demonstration order (default `0`: more similar at top) |
| `--entity_info` | Entity-aware sentence similarity (from our paper) |

### Advanced Options (can be ignored)
| Option | Description |
|--------|-------------|
| `--var` | Keep `0` |
| `--verbalize` | Keep `0` |
| `--structure` | Structured prompt trial, keep default |
| `--use_ft` | Fine-tuned representation - **see [GPT-RE-FT](https://github.com/HanPT831/GPT-RE-FT) for full support** |
| `--self_error` | Keep `0` |
| `--use_dev`, `--store_error_reason`, `--discriminator` | Keep `0` |

## Datasets

The repository includes preprocessed data for:
- SemEval 2010 Task 8
- ACE05
- SciERC

## Citation

If you find this work helpful, please cite our paper:

```bibtex
@inproceedings{wan2023gpt,
  title={GPT-RE: In-context Learning for Relation Extraction using Large Language Models},
  author={Wan, Zhen and Cheng, Fei and Mao, Zhuoyuan and Liu, Qianying and Song, Haiyue and Li, Jiwei and Kurohashi, Sadao},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023}
}
```

## License

Please refer to the LICENSE file for details.


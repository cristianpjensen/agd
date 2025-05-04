# Adapter Guidance Distillation ([📝 Paper](https://arxiv.org/abs/2503.07274))

![Teaser images](docs/teaser.png)

* Adapter Guidance Distillation (AGD) folds classifier-free guidance (CFG) into the model using adapters, so each diffusion step needs only one forward pass, effectively doubling inference speed, while in some cases slightly outperforming CFG.
* The adapters are lightweight and only add 1–5% additional parameters, while keeping the base weights frozen. This also means that they can be disabled to return to the original CFG.
* It is possible to distill CFG of Stable Diffusion XL (2.6B parameters) on a 24 GB VRAM consumer GPU.
* AGD is composable with other methods, such as IP-Adapter and ControlNet.
* No original datasets required—training is done only with samples from the base model.

## Usage

See [the paper](https://arxiv.org/abs/2503.07274) for details on the method and experiments. All commands have a `--help` flag to show the available options.

![Overview of AGD components](docs/overview.png)

### Sampling training trajectories

Diffusion Transformer:
```bash
python -m agd.dit.sample_trajectories --output-dir /path/to/trajectories
```

Stable Diffusion 2.1:
```bash
python -m agd.sd.sample_trajectories --output-dir /path/to/trajectories --base-model stabilityai/stable-diffusion-2-1 --prompt-file prompts/coco2017_train_subset.txt --inference-steps 999
```

Stable Diffusion XL:
```bash
python -m agd.sd.sample_trajectories --output-dir /path/to/trajectories --base-model stabilityai/stable-diffusion-xl-base-1.0 --prompt-file prompts/coco2017_train_subset.txt --inference-steps 1000
```

### Training adapters

Diffusion Transformer:
```bash
python -m agd.dit.train --dir /path/to/results --data-path /path/to/trajectories <...options>
```

Stable Diffusion 2.1:
```bash
python -m agd.sd.train --dir /path/to/results --data-path /path/to/trajectories --base-model stabilityai/stable-diffusion-2-1 <...options>
```

Stable Diffusion XL:
```bash
python -m agd.sd.train --dir /path/to/results --data-path /path/to/trajectories --base-model stabilityai/stable-diffusion-xl-base-1.0 <...options>
```

### Sampling with adapters

Diffusion Transformer:
```bash
python -m agd.dit.sample --dir /path/to/results <...options>
```

Stable Diffusion 2.1/XL:
```bash
python -m agd.sd.sample --dir /path/to/results --prompt <prompt> <...options>
```

### Calculating metrics

Diffusion Transformer:
```bash
python -m agd.dit.calculate_metrics --dir /path/to/results --ref /path/to/ref_samples <...options>
```

Stable Diffusion 2.1/XL:
```bash
python -m agd.sd.calculate_metrics --dir /path/to/results --ref /path/to/ref_samples <...options>
```

## Acknowledgement

This codebase builds upon [the Diffusion Transformer repository](https://github.com/facebookresearch/DiT) and [the diffusers library](https://github.com/huggingface/diffusers).

## Citation

```bib
@article{jensen2025efficient,
  title={Efficient Distillation of Classifier-Free Guidance using Adapters},
  author={Jensen, Cristian Perez and Sadat, Seyedmorteza},
  journal={arXiv preprint arXiv:2503.07274},
  year={2025}
}
```

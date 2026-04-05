# 🎨 Disney GAN — Live Action to Disney Style Transfer

Cartoonización de películas live-action de Disney usando fine-tuning sobre GANs preentrenadas. Este proyecto implementa tres pipelines de style transfer: **Live Action → Disney** usando AnimeGANv2, **Sketch → Disney** usando APDrawingGAN, y un experimento adicional con **CycleGAN**.

---

## Objective

The goal of this project is to automatically transform live-action footage into a Disney animation style using Generative Adversarial Networks. The intended use case is to allow actors to record a scene and pass it through the model instead of animating it by hand.

---

## Project Structure
```
Disney_GAN/
├── README.md
├── LICENSE
├── config/                                        ← Training configuration files
├── data/
│   └── About_Dataset.md                           ← Dataset documentation
├── docs/                                          ← Project report
├── notebooks/
│   ├── 0.0_Dataset_Generator.ipynb                ← Live action dataset pipeline
│   ├── 0.1_Detect_Faces.ipynb                     ← Face detection & alignment
│   ├── 0.2_Dataset_Generator_Sketches_and_Faces.ipynb ← Sketch & cartoon dataset
│   ├── 1.0_AnimeGAN_Fine_Tunning.ipynb            ← AnimeGANv2 fine-tuning
│   ├── 1.1_AnimeGAN_Inference.ipynb               ← AnimeGANv2 inference & ensemble
│   ├── 2.0_APDrawGAN_Fine_Tunning.ipynb           ← APDrawingGAN fine-tuning
│   ├── 2.1_APDrawGAN_Inference.ipynb              ← APDrawingGAN inference
│   └── 3.0_Cycle_Gan_inference.ipynb              ← CycleGAN inference experiment
└── src/
    ├── models/
    │   ├── AnimeGANv2_Generator.py                ← AnimeGANv2 generator architecture
    │   ├── CycleGAN_Generator.py                  ← CycleGAN generator architecture
    │   ├── Discriminator.py                       ← Discriminator architecture
    │   └── Generator.py                           ← Base generator
    ├── utils/
    │   ├── dataset.py                             ← Dataset loading utilities
    │   ├── loss.py                                ← Loss functions
    │   ├── networks.py                            ← Network helpers
    │   └── transformations.py                     ← Image transforms
    ├── visualization/                             ← Generated result images
    └── Weights/
        ├── GeneratorV2_live_action_cartoon_textures.pt  ← AnimeGAN texture model
        ├── GeneratorV2_live_action_cartoon_color.pt     ← AnimeGAN color model v1
        ├── GeneratorV2_live_action_cartoon_color_2.pt   ← AnimeGAN color model v2
        ├── GeneratorV2_gldv2_Hayao.pt                   ← Pretrained Hayao weights
        ├── trained_netG.pth                             ← APDrawingGAN generator
        ├── trained_netD.pth                             ← APDrawingGAN discriminator
        └── Cycle_gan_final.pth                          ← CycleGAN weights
```

---

## Models

### Pipeline 1 — Live Action → Disney (AnimeGANv2)
Takes a real face photo and transforms it into Disney animation style. Uses an ensemble of two fine-tuned generators — one optimized for texture/stylization and one for color accuracy — combined via LAB color space transfer to preserve textures from the texture model while applying the color palette from the color model.

### Pipeline 2 — Sketch → Disney (APDrawingGAN)
Takes a sketch or line drawing and generates a colored Disney-style image using a UNet-based conditional GAN trained on paired sketch-cartoon data.

### Pipeline 3 — CycleGAN & Pix2Pix (Experiment)
Unpaired image-to-image translation experiment using CycleGAN and Pix2Pix as an alternative approach to the live action → Disney transfer without requiring paired training data.

---

## Demo

Try the live demo on Hugging Face Spaces:

👉 [Disney GAN — Hugging Face Space](https://huggingface.co/spaces/pipo1313/Disney_GAN)

---

## Installation
```bash
git clone https://github.com/denisevaldivia/Disney_GAN.git
cd Disney_GAN
```

---

## Evaluation

The models were evaluated using distribution-based metrics suitable for unpaired style transfer:
- FID
- KID
- LPIPS

While the paired data were vealuated with:
- FID
- SSIM

These metrics are expected to be high given the fundamental geometric difference between live-action faces and Disney cartoon characters, and the relatively small dataset size (~500 images per domain). See the full analysis in `docs/`.

---

## Dataset

See [`data/About_Dataset.md`](data/About_Dataset.md) for details on how the dataset was built, including the face detection and alignment pipeline using YOLOv8 and the sketch generation process.

---

## Built With

- [AnimeGANv2](https://github.com/ptran1203/pytorch-animeGAN)
- [APDrawingGAN](https://github.com/yiranran/APDrawingGAN)
- [CycleGAN / Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Gradio](https://gradio.app)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

# VODiff: Controlling Object Visibility Order in Text-to-Image Generation

📄 [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_VODiff_Controlling_Object_Visibility_Order_in_Text-to-Image_Generation_CVPR_2025_paper.pdf)  
🌐 [Project Website](https://dliang293.github.io/vodiff-page/)  

---

Code and project page for our paper **"VODiff: Controlling Object Visibility Order in Text-to-Image Generation"**, accepted to **CVPR 2025**.

---

## ✨ Overview

**VODiff** is a **training-free** framework that introduces **object visibility order** as a new controllable dimension in layout-to-image text-to-image (T2I) generation.

Compared with previous methods, which cannot explicitly control object occlusion, VODiff enables accurate generation of complex scenes with user-defined spatial and occlusion relationships via two core designs:

- **Sequential Denoising Process (SDP):** Synthesizes objects in layers, bottom to top, according to visibility order.
- **Visibility-Order-Aware (VOA) Loss:** Optimizes cross-attention maps to enforce correct spatial and occlusion constraints.

---

## 🖼️ Teaser

![VODiff teaser](https://dliang293.github.io/vodiff-project/VODiff_files/teaser.png)

---

## 🛠️ Environment Setup

```bash
# Create a new conda environment
conda create --name vodiff python=3.9
conda activate vodiff

# Install dependencies
conda install -r requirements.txt
```

---

## 📦 Pretrained Models

Download the pretrained model (e.g., GLIGEN) and place it in the `checkpoints` directory.

---

## 🚀 Inference

To run inference, use the provided Jupyter notebook, please modify these parts to define your own inputs.

```python
caption = 'A car and a bike in front of a house.'
names_list = ['house', 'car', 'bike']  # Ordered by visibility (back to front)
layout = [(66, 197, 452, 390), (326, 358, 402, 432), (111, 347, 216, 431)]  # Corresponding bounding boxes
```

> The `names_list` should be ordered by **visibility**, i.e., from background to foreground.

---

## 🙏 Acknowledgments

This project is built upon the following resources:

- **Attention Refocusing:** Our codebase is based on the foundational work provided by [Attention Refocusing](https://github.com/Attention-Refocusing/attention-refocusing).

If you have any questions or issues, please feel free to open an issue or contact us.

---

## 🪪 License

This project is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license.

🔗 [License Details](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

---

## 📚 Citation

If you find VODiff useful in your research, please consider citing us:

```bibtex
@inproceedings{liang2025vodiff,
  title={VODiff: Controlling Object Visibility Order in Text-to-Image Generation},
  author={Liang, Dong and Jia, Jinyuan and Liu, Yuhao and Ke, Zhanghan and Fu, Hongbo and Lau, Rynson WH},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18379--18389},
  year={2025}
}
```

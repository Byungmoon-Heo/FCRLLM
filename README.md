# FCRLLM
Official source code for WWW 2026 paper "FCRLLLM: Aligning LLM with Collaborative Filtering for Long-tailed Sequential Recommendation"

<br/>

## Environment Setting:

- numpy==1.24.4
- pandas==2.2.2
- scikit-learn==1.5.1
- scipy==1.14.0
- torch==2.5.0
- tqdm==4.66.5
- joblib==1.4.2
<br>

## For dataset
For more detail information about dataset, please refer to [LLM-ESR(https://github.com/Applied-Machine-Learning-Lab/LLM-ESR)].
<br>
After you get datasets, place the file in the `data` folder.

<br/>

## Train Code:
```bash
bash ./experiments/beauty.bash
```
<br/>

## Inference Code:
```bash
bash ./experiments/beauty_inference.bash
```
<br/>

## Reference:
Please cite our paper if you use this code.
```bibtex
@inproceedings{heo2026fcrllm,
  author = {Heo, Byungmoon and Lee, Namjun and Kim, Seonah and Kim, Jaekwang},
  title = {FCRLLM: Aligning LLM with Collaborative Filtering for Long-tailed Sequential Recommendation},
  booktitle = {Proceedings of the ACM Web Conference (WWW)},
  year = {2026},
  pages = {6574--6585},
  doi = {10.1145/3774904.3792565}
}

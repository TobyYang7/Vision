---
pipeline_tag: text-generation
---

## MiniCPM-V
**MiniCPM-V** (i.e., OmniLMM-3B) is an efficient version with promising performance for deployment. The model is built based on SigLip-400M and [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/), connected by a perceiver resampler. Notable features of OmniLMM-3B include:

- ⚡️ **High Efficiency.** 

  MiniCPM-V can be **efficiently deployed on most GPU cards and personal computers**, and **even on end devices such as mobile phones**. In terms of visual encoding, we compress the image representations into 64 tokens via a perceiver resampler, which is significantly fewer than other LMMs based on MLP architecture (typically > 512 tokens). This allows OmniLMM-3B to operate with **much less memory cost and higher speed during inference**.

- 🔥 **Promising Performance.** 

  MiniCPM-V achieves **state-of-the-art performance** on multiple benchmarks (including MMMU, MME, and MMbech, etc) among models with comparable sizes, surpassing existing LMMs built on Phi-2. It even **achieves comparable or better performance than the 9.6B Qwen-VL-Chat**.

- 🙌 **Bilingual Support.** 

  MiniCPM-V is **the first end-deployable LMM supporting bilingual multimodal interaction in English and Chinese**. This is achieved by generalizing multimodal capabilities across languages, a technique from the ICLR 2024 spotlight [paper](https://arxiv.org/abs/2308.12038).

### Evaluation

<div align="center">

<table style="margin: 0px auto;">
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th>MME</th>
    <th nowrap="nowrap" >MMB dev (en)</th>
    <th nowrap="nowrap" >MMB dev (zh)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >CMMMU val</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">LLaVA-Phi</td>
    <td align="right">3.0B</td>
    <td>1335</td>
    <td>59.8</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">MobileVLM</td>
    <td align="right">3.0B</td>
    <td>1289</td>
    <td>59.6</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Imp-v1</td>
    <td align="right">3B</td>
    <td>1434</td>
    <td>66.5</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right" >9.6B</td>
    <td>1487</td>
    <td>60.6 </td>
    <td>56.7 </td>
    <td>35.9 </td>
    <td>30.7 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >CogVLM</td>
    <td align="right">17.4B </td>
    <td>1438 </td>
    <td>63.7 </td>
    <td>53.8 </td>
    <td>32.1 </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>MiniCPM-V</b></td>
    <td align="right">3B </td>
    <td>1452 </td>
    <td>67.9 </td>
    <td>65.3 </td>
    <td>37.2 </td>
    <td>32.1 </td>
  </tr>
</tbody>
</table>

</div>


### Examples
<div align="center">
<table>
  <tr>
    <td>
      <p> 
        <img src="assets/Mushroom_en.gif" width="400"/>
      </p>
    </td>
    <td>
      <p> 
        <img src="assets/Snake_en.gif" width="400"/>
      </p>
    </td>
  </tr>
</table>
</div>


## Demo
Click here to try out the Demo of [MiniCPM-V](http://120.92.209.146:80).

## Deployment on Mobile Phone
Currently MiniCPM-V (i.e., OmniLMM-3B) can be deployed on mobile phones with Android and Harmony operating systems. 🚀 Try it out [here](https://github.com/OpenBMB/mlc-MiniCPM).


## Usage
Inference using Huggingface transformers on Nivdia GPUs or Mac with MPS (Apple silicon or AMD GPUs). Requirements tested on python 3.10：
```
Pillow==10.1.0
timm==0.9.10
torch==2.1.2
torchvision==0.16.2
transformers==4.36.0
sentencepiece==0.1.99
```

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
model = model.to(device='cuda', dtype=torch.bfloat16)
# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
#model = model.to(device='cuda', dtype=torch.float16)
# For Mac with MPS (Apple silicon or AMD GPUs).
# Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py`
#model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

image = Image.open('xx.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question}]

res, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7
)
print(res)
```

Please look at [GitHub](https://github.com/OpenBMB/OmniLMM) for more detail about usage.

## License

#### Model License
* The code in this repo is released according to [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE)
* The usage of MiniCPM-V's parameters is subject to ["General Model License Agreement - Source Notes - Publicity Restrictions - Commercial License"](https://github.com/OpenBMB/General-Model-License/blob/main/)
* The parameters are fully open to acedemic research
* Please contact cpm@modelbest.cn to obtain a written authorization for commercial uses. Free commercial use is also allowed after registration.

#### Statement
* As a LLM, MiniCPM-V generates contents by learning a large mount of texts, but it cannot comprehend, express personal opinions or make value judgement. Anything generated by MiniCPM-V does not represent the views and positions of the model developers
* We will not be liable for any problems arising from the use of the MinCPM-V open Source model, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.

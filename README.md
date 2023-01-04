# waifu-v1-4使ってみたメモ.

## 最低限推論環境ライブラリ
```
pip install git+https://github.com/huggingface/diffusers 
pip install torch==1.13.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers
pip install accelerate
```

推論
```
python waifu1-4.py
```
※ GTX 3070 (VRAM 8GB)でも、２つのオプションをいれることで、768×1024の推論が可能. (5〜6GB程度の使用率.)
```
pipe.enable_attention_slicing() 
pipe.enable_sequential_cpu_offload() 
```

![hoge](000.png)

## 参考ページ

### 推論時のメモリ節約方法.
https://huggingface.co/docs/diffusers/optimization/fp16
https://baskmedia.jp/novelai-code-of-the-elements1/

### diffuserの解説
https://huggingface.co/blog/stable_diffusion
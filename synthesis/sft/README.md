## Guidance of Implementaion and Debuging
输入一个问题集合，模型回答并将完整CoT打印到终端：
```bash
python -m synthesis.sft.debug_vqa_batch \
  --vqa-dir /abs/path/to/vqa_dir \
  --model your-answer-model \
  --expert-model your-expert-model \
  --limit 5
```

# QLoRA Fine-tuning Plan

Version 1 of ClaimMate AI does not require fine-tuning.

Fine-tuning is planned only for structured response behavior, not for memorizing warranty policies.

Train the model to:
- classify warranty/return/service situations
- produce structured claim analysis
- detect missing documents
- draft polite truthful support emails
- refuse fake claim or fraud requests

Starter dataset:

```text
data/qlora_sft_claimmate.jsonl
```

A future `train_qlora.py` can use TRL `SFTTrainer`, PEFT, and bitsandbytes depending on GPU availability.

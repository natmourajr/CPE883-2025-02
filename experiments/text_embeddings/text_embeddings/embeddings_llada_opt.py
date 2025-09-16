"""
Use this module ONLY with the correct environment for ModernBERT.
"""
import torch
from transformers import AutoModel, AutoTokenizer

from .embeddings import EmbeddingExtractor


class LLaDA_8B_Extractor(EmbeddingExtractor):

    def __init__(self,
        model_path: str= 'GSAI-ML/LLaDA-8B-Base',
        bos_token: str = '[CLS]',
        eos_token: str = '[SEP]',
        mask_token: str = '[MASK]',
    ):
        super().__init__(
            model_path=model_path,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
        )

    def load_model(self):
        # NO-OP if model is already loaded
        if self.model is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map='cuda')

        self.tokenizer = tokenizer
        self.model = model
        assert self.tokenizer is not None

        self.bos_token = tokenizer.special_tokens_map['bos_token']
        self.eos_token = tokenizer.special_tokens_map['eos_token']
        self.mask_token = tokenizer.special_tokens_map['additional_special_tokens'][0]

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        self.max_length = self.model.config.max_sequence_length
        self.embedding_size = self.model.config.embedding_size

    def eval_embeddings(self, inputs, mask_idxs) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
        mask_idxs : torch.Tensor shape (batch_size,)

        Returns
        -------
        torch.Tensor shape (batch_size, hidden_size)
        """
        assert self.model is not None
        assert self.tokenizer is not None

        with torch.no_grad():
            outputs = self.model.base_model(
                input_ids=inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
            )

        state = outputs.logits.cpu().float()

        batch_idx = torch.arange(state.size(0))
        return state[batch_idx, mask_idxs, :]

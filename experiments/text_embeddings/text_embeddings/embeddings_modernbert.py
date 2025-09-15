"""
Use this module ONLY with the correct environment for ModernBERT.
"""
import torch
from composer.models.huggingface import HuggingFaceModel

from .embeddings import EmbeddingExtractor


class ComposerExtractor(EmbeddingExtractor):

    def __init__(self,
        model_path: str,
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

        model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint(self.model_path)
        model = model.to('cuda')


        self.tokenizer = tokenizer
        assert self.tokenizer is not None

        self.model = model

        self.mask_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        self.max_length = self.model.config.max_position_embeddings
        self.embedding_size = self.model.config.hidden_size

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

        state = outputs.cpu()

        batch_idx = torch.arange(state.size(0))
        return state[batch_idx, mask_idxs, :]

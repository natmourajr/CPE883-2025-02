"""
Derive embeddings from models
"""
from abc import ABC, abstractmethod

from transformers import AutoModel, AutoTokenizer, BertForMaskedLM, PreTrainedTokenizerBase 
import torch


class EmbeddingExtractor(ABC):

    def __init__(self,
        model_path: str,
        bos_token: str = '[BOS]',
        eos_token: str = '[EOS]',
        mask_token: str = '[MASK]',
    ):
        self.model = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        self.model_path = model_path

        self.max_length = 512
        self.mask_id: int = 0
        self.embedding_size: int = 0

    @abstractmethod
    def load_model(self):
        pass

    def format_text_prompteol(self, text: str):

        assert self.tokenizer is not None

        toks = self.tokenizer.tokenize(text)

        token_lim = 475

        if len(toks) > token_lim:
            high = len(text)
            low = high

            iter = 0

            while True:
                iter += 1
                if iter > 1000:
                    print('Loop Inifinito 1')
                    print(f'{text=}')
                    print(f'{low=}')
                    print(f'{high=}')
                    break

                low = low // 2
                curr = len(self.tokenizer.tokenize(text[:low]))
                if curr < token_lim:
                    break

            iter = 0
            while True:
                iter += 1
                if iter > 1000:
                    print('Loop Inifinito 2')
                    print(f'{text=}')
                    print(f'{low=}')
                    print(f'{high=}')
                    break
                mid = (low + high) // 2
                curr = len(self.tokenizer.tokenize(text[:mid]))
                if curr <= token_lim:
                    low = curr
                else:
                    high = curr

                if low - high <= 10:
                    break

            text = text[:low]

        return (
            f'{self.bos_token} This text: " {text} " means in one word :'
            f' " {self.mask_token} " . {self.eos_token}'
        )

    @abstractmethod
    def eval_embeddings(self, inputs, mask_idxs) -> torch.Tensor:
        pass

    def get_embeddings_batch(self, batch: list[str],) -> torch.Tensor:
        self.load_model()
        formatted = [self.format_text_prompteol(x) for x in batch]

        assert self.tokenizer is not None

        inputs = self.tokenizer(
            formatted,
            padding=True,
            max_length=512,
            return_tensors='pt',
            return_overflowing_tokens=False,
        )
        rows, cols = torch.where(inputs['input_ids'] == self.mask_id)

        max_cols = torch.scatter_reduce(
            torch.full((rows.max() + 1,), 0, dtype=cols.dtype),
            0, rows, cols, reduce='amax',
        )

        return self.eval_embeddings(inputs, max_cols)

    def mask_token_inference(self, text: str, k=5) -> list[tuple[str, str]]:
        self.load_model()

        assert self.tokenizer is not None
        assert self.model is not None

        inputs = self.tokenizer(
            [text],
            padding=True,
            max_length=512,
            return_tensors='pt',
            return_overflowing_tokens=False,
        )

        mask_idx = torch.where(inputs['input_ids'] == self.mask_id)[1].item()

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
            )

        logits = outputs.logits.cpu()

        mask_logits = logits[0, mask_idx, :]
        topk = torch.topk(mask_logits, k=k)

        tokens = self.tokenizer.convert_ids_to_tokens(topk.indices.tolist())

        scores = topk.values.tolist()

        results = list(zip(tokens, scores))
        return results


class HFBertExtractor(EmbeddingExtractor):

    def __init__(self,
        model_path: str,
        bos_token: str = '[BOS]',
        eos_token: str = '[EOS]',
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        assert self.tokenizer is not None

        self.model = BertForMaskedLM.from_pretrained(self.model_path,
                                                     device_map='cuda')

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

        state = outputs.last_hidden_state.cpu()

        batch_idx = torch.arange(state.size(0))
        return state[batch_idx, mask_idxs, :]


class HFAutoExtractor(EmbeddingExtractor):

    def __init__(self,
        model_path: str,
        bos_token: str = '[BOS]',
        eos_token: str = '[EOS]',
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        assert self.tokenizer is not None

        self.model = AutoModel.from_pretrained(self.model_path,
                                               device_map='cuda')

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

        state = outputs.last_hidden_state.cpu()

        batch_idx = torch.arange(state.size(0))
        return state[batch_idx, mask_idxs, :]

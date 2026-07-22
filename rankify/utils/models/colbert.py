import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer
from typing import List, Optional, Union
from math import ceil




def _insert_token(
    output: dict,
    insert_token_id: int,
    insert_position: int = 1,
    token_type_id: int = 0,
    attention_value: int = 1,
):
    """
    Inserts a new token at a specified position into the sequences of a tokenized representation.

    This function takes a dictionary containing tokenized representations
    (e.g., 'input_ids', 'token_type_ids', 'attention_mask') as PyTorch tensors,
    and inserts a specified token into each sequence at the given position.
    This can be used to add special tokens or other modifications to tokenized inputs.

    Parameters:
    - output (dict): A dictionary containing the tokenized representations. Expected keys
                     are 'input_ids', 'token_type_ids', and 'attention_mask'. Each key
                     is associated with a PyTorch tensor.
    - insert_token_id (int): The token ID to be inserted into each sequence.
    - insert_position (int, optional): The position in the sequence where the new token
                                       should be inserted. Defaults to 1, which typically
                                       follows a special starting token like '[CLS]' or '[BOS]'.
    - token_type_id (int, optional): The token type ID to assign to the inserted token.
                                     Defaults to 0.
    - attention_value (int, optional): The attention mask value to assign to the inserted token.
                                        Defaults to 1.

    Returns:
    - updated_output (dict): A dictionary containing the updated tokenized representations,
                             with the new token inserted at the specified position in each sequence.
                             The structure and keys of the output dictionary are the same as the input.
    """
    updated_output = {}
    for key in output:
        updated_tensor_list = []
        for seqs in output[key]:
            if len(seqs.shape) == 1:
                seqs = seqs.unsqueeze(0)
            for seq in seqs:
                first_part = seq[:insert_position]
                second_part = seq[insert_position:]
                new_element = (
                    torch.tensor([insert_token_id])
                    if key == "input_ids"
                    else torch.tensor([token_type_id])
                )
                if key == "attention_mask":
                    new_element = torch.tensor([attention_value])
                updated_seq = torch.cat((first_part, new_element, second_part), dim=0)
                updated_tensor_list.append(updated_seq)
        updated_output[key] = torch.stack(updated_tensor_list)
    return updated_output


def _colbert_score(q_reps, p_reps, q_mask: torch.Tensor, p_mask: torch.Tensor):
    # calc max sim
    # base code from: https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/BGE_M3/modeling.py

    # Assert that all q_reps are at least as long as the query length
    assert (
        q_reps.shape[1] >= q_mask.shape[1]
    ), f"q_reps should have at least {q_mask.shape[1]} tokens, but has {q_reps.shape[1]}"

    token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
    token_scores = token_scores.masked_fill(p_mask.unsqueeze(0).unsqueeze(0) == 0, -1e4)
    scores, _ = token_scores.max(-1)
    scores = scores.sum(1) / q_mask.sum(-1, keepdim=True)
    return scores


class ColBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        # TODO: Load from artifact.metadata
        if "small" in config._name_or_path:
            linear_dim = 96
        else:
            linear_dim = 128
        #print(f"Linear Dim set to: {linear_dim} for downcasting")
        self.linear = nn.Linear(config.hidden_size, linear_dim, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always output hidden states
        )

        sequence_output = outputs[0]

        return self.linear(sequence_output)

    def _encode(self, texts: list[str], insert_token_id: int, is_query: bool = False):
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length - 1,  # for insert token
            truncation=True,
        )
        encoding = _insert_token(encoding, insert_token_id)  # type: ignore

        if is_query:
            mask_token_id = self.tokenizer.mask_token_id

            new_encodings = {"input_ids": [], "attention_mask": []}

            for i, input_ids in enumerate(encoding["input_ids"]):
                original_length = (
                    (input_ids != self.tokenizer.pad_token_id).sum().item()
                )

                # Calculate QLEN dynamically for each query
                if original_length % 32 <= 8:
                    QLEN = original_length + 8
                else:
                    QLEN = ceil(original_length / 32) * 32

                if original_length < QLEN:
                    pad_length = QLEN - original_length
                    padded_input_ids = input_ids.tolist() + [mask_token_id] * pad_length
                    padded_attention_mask = (
                        encoding["attention_mask"][i].tolist() + [0] * pad_length
                    )
                else:
                    padded_input_ids = input_ids[:QLEN].tolist()
                    padded_attention_mask = encoding["attention_mask"][i][
                        :QLEN
                    ].tolist()

                new_encodings["input_ids"].append(padded_input_ids)
                new_encodings["attention_mask"].append(padded_attention_mask)

            for key in new_encodings:
                new_encodings[key] = torch.tensor(
                    new_encodings[key], device=self.device
                )

            encoding = new_encodings

        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        return encoding

    def _query_encode(self, query: list[str]):
        return self._encode(query, self.query_token_id, is_query=True)

    def _document_encode(self, documents: list[str]):
        return self._encode(documents, self.document_token_id)

    def _to_embs(self, encoding) -> torch.Tensor:
        with torch.inference_mode():
            # embs = self.model(**encoding).last_hidden_state.squeeze(1)
            embs = self.model(**encoding)
        if self.normalize:
            embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs

    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_encoding = self._query_encode([query])
        documents_encoding = self._document_encode(documents)
        query_embeddings = self._to_embs(query_encoding)
        document_embeddings = self._to_embs(documents_encoding)
        scores = (
            _colbert_score(
                query_embeddings,
                document_embeddings,
                query_encoding["attention_mask"],
                documents_encoding["attention_mask"],
            )
            .cpu()
            .tolist()[0]
        )
        return scores
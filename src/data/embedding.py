import torch
from transformers import BertTokenizer, BertModel

from common.config import data_dir


# reference: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
class BertEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            cache_dir=f"{data_dir}/HuggingFace"
        )
        self.model = BertModel.from_pretrained(
            'bert-base-uncased',
            # Whether the model returns all hidden-states.
            output_hidden_states=True,
            cache_dir=f"{data_dir}/HuggingFace")
        self.model.eval()

    def get_embedding(self, text):
        text = f"[CLS] {text} [SEP]"
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segment_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segment_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensor)
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]

        return torch.mean(token_vecs, dim=0)


class FastTextEmbedding:
    def __init__(self, seq_len, pad=True):
        self.ft = fasttext.load_model(f'{data_dir}/FastText/cc.en.300.bin')
        self.seq_len = seq_len

    def get_embedding(self, text):
        embed_seq = []
        for i, word in enumerate(text):
            embed = self.ft[word]
            embed_seq.append(embed)

        while i < self.seq_len:
            embed_seq.append(torch.zeros(300,))
            i += 1

        return torch.FloatTensor(embed_seq)

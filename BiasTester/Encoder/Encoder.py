import torch
import numpy as np
from transformers import BertForNextSentencePrediction, AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from BiasTester.Encoder.EncoderClass import Encoder
from sentence_transformers import SentenceTransformer


class TokenizationError(Exception):
    pass


class BertCLSEncoder(Encoder):
    def __init__(self, version='bert-large-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModel.from_pretrained(version)
        self.model.eval()

    def documents_to_vecs(self, sentences: [str], mode="CLS", obj=None):
        #print(sentences)
        #print(obj)
        #encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        encoded_input = self.tokenizer(sentences, padding=True, return_tensors="pt")
        #print(encoded_input)
        with torch.no_grad():
            output = self.model(**encoded_input)

        results = []
        for sent_idx, sent in enumerate(output["last_hidden_state"]):  #Kann man garantiert auch eleganter lösen ....
            if mode == "CLS":
                results.append(sent[0])
            elif mode == "AVG":
                results.append(torch.tensor(np.average(sent.numpy(), 0)))
            elif mode == "MAX":
                results.append(torch.tensor(np.max(sent.numpy(), 0)))
            elif mode == "OBJ":
                #print(obj)
                encoded_obj = self.tokenizer.encode(obj, add_special_tokens=False)
                encoded_sent = encoded_input["input_ids"][sent_idx]
                obj_results = []

                for enc_obj in encoded_obj:
                    obj_id = (encoded_sent == enc_obj).nonzero(as_tuple=True)[0][0] #Find for every objid the right index in sent
                    obj_results.append(sent[obj_id].numpy())

                obj_results = np.array(obj_results)
                results.append(torch.tensor(np.average(obj_results, 0)))
        results = torch.stack(results, 0)
        return results

    #For Decoder ...
    def document_to_vec(self, sentences: [str], mode="CLS", obj=None):
        results = []
        for sent in sentences:
            encoded_input = self.tokenizer(sent, truncation=True, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**encoded_input)["last_hidden_state"][0]

            if mode == "CLS":
                results.append(output[0])
            elif mode == "AVG":
                results.append(torch.tensor(np.average(output.numpy(), 0)))
            elif mode == "MAX":
                results.append(torch.tensor(np.max(output.numpy(), 0)))
            elif mode == "OBJ":
                #print(obj)
                encoded_obj = self.tokenizer.encode(obj, add_special_tokens=False)
                encoded_sent = encoded_input["input_ids"][0]
                obj_results = []

                for enc_obj in encoded_obj:
                    obj_id = (encoded_sent == enc_obj).nonzero(as_tuple=True)[0][0] #Find for every objid the right index in sent
                    obj_results.append(output[obj_id].numpy())

                obj_results = np.array(obj_results)
                results.append(torch.tensor(np.average(obj_results, 0)))
        results = torch.stack(results, 0)
        return results

    def sent_encoder(self, sent):
        encoded_input = self.tokenizer(sent, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output


class BertMASKEncoder(Encoder):
    def __init__(self, version='bert-large-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModelForMaskedLM.from_pretrained(version, return_dict=True)
        self.model.eval()


class BertCausalEncoder(Encoder):
    def __init__(self, version='gpt2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModelForCausalLM.from_pretrained(version, return_dict=True)
        self.model.eval()


class BertPoolerEncoder(Encoder):
    def __init__(self, version='bert-large-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = AutoModel.from_pretrained(version)

    def documents_to_vecs(self, sentences: [str]):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output["pooler_output"]


class BertNextSentencePred(Encoder):
    def __init__(self, version='bert-large-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.model = BertForNextSentencePrediction.from_pretrained(version)

    def sentence_pred(self, sentence1: str, sentence2: str):
        encoded_input = self.tokenizer(sentence1, sentence2, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output.logits

    def documents_to_vecs(self, sentences: [str]):
        pass


class BertSentenceEncoder(Encoder):
    def __init__(self, version='paraphrase-MiniLM-L6-v2'):
        super().__init__()
        self.model = SentenceTransformer(version)

    def documents_to_vecs(self, sentences: [str]):
        embeddings = self.model.encode(sentences)
        return torch.FloatTensor(embeddings)


if __name__ == "__main__":
    encoder = BertCLSEncoder("roberta-large")
    encod = encoder.documents_to_vecs(["this is a test.", "This is a second test"])
    print(encod)

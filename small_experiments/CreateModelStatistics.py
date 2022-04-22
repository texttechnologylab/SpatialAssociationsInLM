from transformers import AutoModel
from torchinfo import summary

from CreateAllNyuResults import run as run1
from CreateAllPartResults import run as run2
from CreateAllVerbResults import run as run3


def run():
    model_names = ['bert-base-uncased', 'bert-large-uncased', "albert-xxlarge-v2",
                   "roberta-large", "google/electra-large-generator",
                   "gpt2-large", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-j-6B"]

    #model_names = ["roberta-large", "google/electra-large-generator",
    #               "gpt2-large"]

    for model_name in model_names:
        model = AutoModel.from_pretrained(model_name)
        print(model_name)
        #print(summary(model, input_size=(16, 512)))
        summary(model, input_size=(16, 512), dtypes=['torch.IntTensor'], device='cpu')
        #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("==========================")


if __name__ == "__main__":
    #run()
    #run1()
    run2()
    run3()

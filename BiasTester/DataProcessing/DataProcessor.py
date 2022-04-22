from BiasTester.DataProcessing.DataProcessorClass import DataProcessor
from BiasTester.Datasets.BiasDatasetClass import BiasDataset


class WEATProcessor(DataProcessor):
    """
    On Measuring Social Biases in Sentence Encoders: https://arxiv.org/abs/1903.10561
    https://github.com/W4ngatang/sent-bias
    """
    TEMPLATES = [
        'This is {article} {term}.',
        'That is {article} {term}.',
        'There is {article} {term}.',
        'Here is {article} {term}.',
        '{article} {term} is here.',
        '{article} {term} is there.',
    ]

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    def get_templates(self, obj: str):
        return [self.fill_template(template, obj) for template in self.TEMPLATES]


    @staticmethod
    def fill_template(template, term):
        article = 'an' if any(term.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = template.format(article=article, term=term)
        return sentence[0].upper() + sentence[1:]


class WEATVerbProcessor(DataProcessor):
    """
    On Measuring Social Biases in Sentence Encoders: https://arxiv.org/abs/1903.10561
    https://github.com/W4ngatang/sent-bias
    """
    OBJTEMPLATES = [
        'This is {article} {term}.',
        'That is {article} {term}.',
        'There is {article} {term}.',
        'Here is {article} {term}.',
        '{article} {term} is here.',
        '{article} {term} is there.',
    ]

    VERBTEMPLATES = [
        'I {verb} something.',
        'I {verb} anything.',
        'I {verb}.',
        'You {verb} something.',
        'You {verb} anything.',
        'You {verb}.',

    ]

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    def get_obj_templates(self, obj: str):
        return [self.fill_obj_template(template, obj) for template in self.OBJTEMPLATES]

    def get_verb_templates(self, obj: str):
        return [self.fill_verb_template(template, obj) for template in self.VERBTEMPLATES]

    @staticmethod
    def fill_verb_template(template, verb):
        sentence = template.format(verb=verb)
        return sentence[0].upper() + sentence[1:]

    @staticmethod
    def fill_obj_template(template, obj):
        article = 'an' if any(obj.startswith(c) for c in 'aeiouAEIOU') else 'a'
        sentence = template.format(article=article, term=obj)
        return sentence[0].upper() + sentence[1:]


class MaskObjProcessor(DataProcessor):
    TEMPLATE = "{article} {obj} is usually in the {room}."
    #TEMPLATEPART = "The {obj} is part of {article2} {room}."

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(obj, room):
        article = 'an' if any(obj.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = MaskObjProcessor.TEMPLATE.format(obj=obj, room=room, article=article)
        return sentence[0].upper() + sentence[1:]


class MaskObjProcessorI(DataProcessor):
    TEMPLATE = "In the {room} is usually {article} {obj}."

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(obj, room):
        article = 'an' if any(obj.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = MaskObjProcessorI.TEMPLATE.format(obj=obj, room=room, article=article)
        return sentence[0].upper() + sentence[1:]


class MaskPartProcessor(DataProcessor):
    TEMPLATE = "{article} {part} is usually part of {article2} {obj}."

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(part, obj):
        article = 'an' if any(part.startswith(c) for c in 'aeiouAEIOU') else 'a'
        article2 = 'an' if any(obj.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = MaskPartProcessor.TEMPLATE.format(obj=obj, part=part, article2=article2, article=article)
        return sentence[0].upper() + sentence[1:]


class MaskPartVerbProcessor(DataProcessor):
    TEMPLATE = "I usually {verb} this {obj}."

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(obj, verb):
        sentence = MaskPartVerbProcessor.TEMPLATE.format(verb=verb, obj=obj)
        return sentence[0].upper() + sentence[1:]


class PredObjProcessor(DataProcessor):

    TEMPLATE = "{article} {obj} is usually in the"
    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(obj):
        article = 'an' if any(obj.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = PredObjProcessor.TEMPLATE.format(obj=obj,article=article)
        return sentence[0].upper() + sentence[1:]



class PredPartProcessor(DataProcessor):

    TEMPLATEPART = "{article} {obj} is usually part of a"

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(part):
        article = 'an' if any(part.startswith(c) for c in 'aeiouAEIOU') else 'a'

        sentence = PredPartProcessor.TEMPLATEPART.format(obj=part, article=article)
        return sentence[0].upper() + sentence[1:]


class PredObjVerbProcessor(DataProcessor):

    TEMPLATEPART = "I usually {verb} this"

    def __init__(self):
        super().__init__()

    def process(self, dataset: BiasDataset):
        pass

    @staticmethod
    def fill_template(verb):
        sentence = PredPartProcessor.TEMPLATEPART.format(verb=verb)
        return sentence[0].upper() + sentence[1:]

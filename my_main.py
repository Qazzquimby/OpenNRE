import os

import opennre
import typing as t


def start_end_indices(string: str, substring: str) -> t.Optional[t.Tuple[int, int]]:
    start = string.find(substring)
    if start == -1:
        return None
    else:
        return start, start + len(substring)


def infer(text: str, head: str, tail: str, model_id: str = 'wiki80_cnn_softmax'):
    model = opennre.get_model(model_id, root_path=os.path.join(os.getcwd(), 'opennre_downloads'))
    return model.infer({'text': text,
                        'h': {'pos': start_end_indices(text, head)},
                        't': {'pos': start_end_indices(text, tail)}})


if __name__ == '__main__':
    text = 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).'
    head = 'Máel Dúin mac Máele Fithrich'
    tail = 'Áed Uaridnach'

    result = infer(text, head, tail)
    print(result)

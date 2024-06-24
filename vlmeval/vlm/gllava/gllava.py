import traceback

import torch
from PIL import Image
import sys
from ..base import BaseModel
from ...smp import *
from ...utils import DATASET_TYPE


class GLLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_pth='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from eval.model_runner import ModelRunner
        except:
            traceback.print_exc()
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        config_path = os.path.join(model_pth, "model.params")
        load_nbit = 16
        self.runner = ModelRunner(checkpoint_path=model_pth, config_path=config_path, load_nbit=load_nbit)

        self.conv_mode = self.runner.default_prompt_version()
        #do_sample = True, temperature = 0.2, top_p = None, max_new_tokens = 64
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):

        from data.convs.conv_base import DEFAULT_IMAGE_TOKEN
        from data.convs.conv_templates import get_empty_conv

        cur_prompt, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                cur_prompt += msg['value']
            elif msg['type'] == 'image':
                cur_prompt += DEFAULT_IMAGE_TOKEN + '\n'
                images.append(msg['value'])

        images = [Image.open(s).convert('RGB') for s in images]

        # Support interleave text and image
        conv = get_empty_conv(self.conv_mode)
        conv.append_message(conv.roles[0], cur_prompt)
        conv.append_message(conv.roles[1], None)
        prompts = [conv]

        with torch.inference_mode():
            results = self.runner.run(prompts, images, **self.kwargs)
        print(results)
        return results[0]



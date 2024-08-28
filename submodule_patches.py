from torch.utils.data import DataLoader

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


def patch_submodules():
    patch_dataloader()
    patch_llava_forward()


def patch_dataloader():
    """
    Getting this error when run without this patch.

    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/cog/server/worker.py", line 221, in _predict
    result = predict(**payload)
    File "/src/train.py", line 156, in train
    job.run()
    File "/src/ai-toolkit/jobs/ExtensionJob.py", line 22, in run
    process.run()
    File "/src/ai-toolkit/jobs/process/BaseSDTrainProcess.py", line 1595, in run
    dataloader_iterator = iter(dataloader)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 440, in __iter__
    return self._get_iterator()
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 388, in _get_iterator
    return _MultiProcessingDataLoaderIter(self)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1038, in __init__
    w.start()
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/process.py", line 121, in start
    self._popen = self._Popen(self)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/context.py", line 288, in _Popen
    return Popen(process_obj)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    super().__init__(process_obj)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 47, in _launch
    reduction.dump(process_obj, fp)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
    TypeError: cannot pickle 'weakref.ReferenceType' object
    """

    old_init = DataLoader.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        if "prefetch_factor" in kwargs:
            del kwargs["prefetch_factor"]
        old_init(self, *args, **kwargs)

    DataLoader.__init__ = patched_init


def patch_llava_forward():
    """
    Getting this error when run without this patch.

    Traceback (most recent call last):
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/cog/server/worker.py", line 354, in _predict
    result = predict(**payload)
    File "/src/train.py", line 152, in train
    captioner.caption_images(INPUT_DIR, autocaption_prefix, autocaption_suffix)
    File "/src/caption.py", line 121, in caption_images
    self.caption_image(
    File "/src/caption.py", line 167, in caption_image
    output_ids = self.model.generate(
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
    File "/src/LLaVA/llava/model/language_model/llava_llama.py", line 137, in generate
    return super().generate(
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/transformers/generation/utils.py", line 2024, in generate
    result = self._sample(
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/transformers/generation/utils.py", line 2982, in _sample
    outputs = self(**model_inputs, return_dict=True)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
    File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
    TypeError: LlavaLlamaForCausalLM.forward() got an unexpected keyword argument 'cache_position'
    """
    old_forward = LlavaLlamaForCausalLM.forward

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
        # Ignore this argument
        cache_position=None,  # noqa: ARG001
    ):
        return old_forward(
            self,
            input_ids=input_ids,  # pyright: ignore
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            image_sizes=image_sizes,
            return_dict=return_dict,
        )

    LlavaLlamaForCausalLM.forward = patched_forward

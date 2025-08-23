import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    all_wavs = []
    all_melspecs = []
    max_len_wav = 0
    max_len_spec = 0
    paths = []
    initial_lens = []
    initial_melspec_lens = []
    gt_wavs = []
    gt_melspecs = []
    max_len_gt_wav = 0
    max_len_gt_spec = 0
    initial_gt_lens = []
    initial_gt_melspec_lens = []


    for item in dataset_items:
        paths.append(item['path'])
        all_wavs.append(item['wav'].squeeze(0))
        all_melspecs.append(item['melspec'])
        max_len_wav = max(len(item['wav'].squeeze(0)), max_len_wav)
        max_len_spec =  max(item['melspec'].shape[-1], max_len_spec)
        initial_lens.append(item['wav'].shape[1])
        initial_melspec_lens.append(item['melspec'].shape[-1])
        gt_wavs.append(item['gt_wav'].squeeze(0))
        gt_melspecs.append(item['gt_melspec'])
        max_len_gt_wav = max(len(item['gt_wav'].squeeze(0)), max_len_gt_wav)
        max_len_gt_spec =  max(item['gt_melspec'].shape[-1], max_len_gt_spec)
        initial_gt_lens.append(item['gt_wav'].shape[1])
        initial_gt_melspec_lens.append(item['gt_melspec'].shape[-1])

    result_batch['initial_len'] = initial_lens
    result_batch['initial_gt_len'] = initial_gt_lens
    
    padded_wavs = torch.stack([F.pad(wav, (0, max_len_wav - wav.shape[0]), value=0) for wav in all_wavs])
    result_batch['initial_melspec_len'] = initial_melspec_lens
    padded_specs = torch.stack([F.pad(spec, (0, max_len_spec - spec.shape[-1], 0, 0)) for spec in all_melspecs])
    result_batch['wav'] = padded_wavs.unsqueeze(1)
    result_batch['melspec'] = padded_specs


    padded_gt_wavs = torch.stack([F.pad(wav, (0, max_len_gt_wav - wav.shape[0]), value=0) for wav in gt_wavs])
    result_batch['initial_gt_melspec_len'] = initial_gt_melspec_lens
    padded_gt_specs = torch.stack([F.pad(spec, (0, max_len_gt_spec - spec.shape[-1], 0, 0)) for spec in gt_melspecs])
    result_batch['gt_wav'] = padded_gt_wavs.unsqueeze(1)
    result_batch['gt_melspec'] = padded_gt_specs

    result_batch['path'] = paths
    return result_batch
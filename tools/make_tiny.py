"""A tiny random ModernBERT token classifier with a small BPE tokenizer trained on RAGTruth text.

Only for checking that every step runs on a CPU without downloads; its scores mean nothing.

    python tools/make_tiny.py --data-dir RAGTruth --out work_test/tiny
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers  # noqa: E402
from transformers import ModernBertConfig, ModernBertForTokenClassification, PreTrainedTokenizerFast  # noqa: E402

from odh.data import load_ragtruth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    samples = load_ragtruth(args.data_dir)[:3000]
    texts = [s['answer'] for s in samples] + [s['prompt'] for s in samples[::6]]
    tok = Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ['[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]']
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=3000, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    cls, sep = tok.token_to_id('[CLS]'), tok.token_to_id('[SEP]')
    tok.post_processor = processors.TemplateProcessing(
        single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
        special_tokens=[('[CLS]', cls), ('[SEP]', sep)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', cls_token='[CLS]',
                                   sep_token='[SEP]', pad_token='[PAD]', mask_token='[MASK]')
    os.makedirs(args.out, exist_ok=True)
    fast.save_pretrained(args.out)
    cfg = ModernBertConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128,
                           num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=4096,
                           pad_token_id=fast.pad_token_id, cls_token_id=cls, sep_token_id=sep,
                           bos_token_id=cls, eos_token_id=sep, num_labels=2)
    ModernBertForTokenClassification(cfg).save_pretrained(args.out)
    print('saved', args.out)


if __name__ == '__main__':
    main()

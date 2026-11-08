"""Shared checkpoint names and prompts for training and inference."""

MODELS = {
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-3b": dict(hf="Qwen/Qwen2.5-3B-Instruct"),
    "qwen2.5-7b": dict(hf="Qwen/Qwen2.5-7B-Instruct", four_bit=True),
    "hunyuan-mt-7b": dict(hf="tencent/Hunyuan-MT-7B", four_bit=True),
    "tower-7b": dict(hf="Unbabel/TowerInstruct-7B-v0.2", four_bit=True),
    "tower-mistral-7b": dict(hf="Unbabel/TowerInstruct-Mistral-7B-v0.2", four_bit=True,
                            prompt="Translate the following text from Chinese into English.\nChinese: {src}\nEnglish:"),
    "alma-7b": dict(hf="haoranxu/ALMA-7B", four_bit=True, chat=False),
    "madlad400-3b": dict(hf="google/madlad400-3b-mt", seq2seq=True, target_prefix="<2en> ", streaming=True),
    "nllb-1.3b": dict(hf="facebook/nllb-200-distilled-1.3B", seq2seq=True,
                      src_lang="zho_Hans", tgt_lang="eng_Latn"),
}
TEAM_MODELS = tuple(name for name in MODELS if name != "alma-7b")
PROMPT = "Translate the following Chinese text into English. Output only the English translation.\n\nChinese: {src}"
PLAIN_PROMPT = "Translate this from Chinese into English:\nChinese: {src}\nEnglish:"


def build_prompt(tok, cfg, src):
    if cfg.get("seq2seq"):
        return cfg.get("target_prefix", "") + src
    if cfg.get("chat", True) and tok.chat_template:
        return tok.apply_chat_template(
            [{"role": "user", "content": cfg.get("prompt", PROMPT).format(src=src)}],
            tokenize=False, add_generation_prompt=True)
    return PLAIN_PROMPT.format(src=src)

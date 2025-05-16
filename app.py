from flask import Flask, render_template, request
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Import your normalization functions
from normalization import (
    unicodeToAscii,
    REMOVE_BRACKETS_TRANS,
    remove_brackets,
    normalize_digits,
    normalize_brackets,
    gap_filler,
    fix_cuneiform_gap,
    fix_suprasigillum,
    read_and_process_file,
    convert,
    collapse_spaces,
    remove_control_characters,
    normalize,
    normalizeString_cuneiform,
    normalizeString_cuneiform_transliterate_translate,
    normalizeString_en,
    trim_singles
)

app = Flask(__name__)

# Mapping of source languages to their models and prompt options
MODEL_CONFIG = {
    "Akkadian": {
        "model_name": "thalesian/AKK-60m",
        "prompt_styles": {
            "Cuneiform → English": "Translate Akkadian cuneiform to English: ",
            "Transliteration → English": "Translate complex Akkadian transliteration to English: ",
            "English → Cuneiform": "Translate English to Akkadian cuneiform: ",
            "English → Transliteration": "Translate English to complex Akkadian transliteration: ",
            "Cuneiform → Transliteration": "Transliterate Akkadian cuneiform to complex Latin characters: "
        }
    },
    "Hittite": {
        "model_name": "thalesian/HIT-60m",
        "prompt_styles": {
            "Transliteration → English": "Translate Hittite transliteration to English: ",
            "Transliteration → German": "Translate Hittite transliteration to German: ",
            "English → Transliteration": "Translate English to Hittite transliteration: ",
            "German → Transliteration": "Translate German to Hittite transliteration: "
        }
    }
}

# Caches for loaded models and tokenizers
_models = {}
_tokenizers = {}

def get_model_and_tokenizer(source):
    cfg = MODEL_CONFIG[source]
    if source not in _models:
        _tokenizers[source] = T5Tokenizer.from_pretrained(cfg["model_name"], use_fast=False)
        _models[source] = T5ForConditionalGeneration.from_pretrained(cfg["model_name"])
    return _models[source], _tokenizers[source]

@app.route('/', methods=['GET', 'POST'])
def index():
    # Determine selected source and prompt style
    source = request.form.get('source_lang', 'Akkadian')
    prompt_key = request.form.get('prompt_style')
    translation = None

    # Prepare prompt mapping for frontend
    prompt_map = { src: list(cfg['prompt_styles'].keys()) for src, cfg in MODEL_CONFIG.items() }

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(source)

    # Build dropdown options
    sources = list(MODEL_CONFIG.keys())
    prompt_styles = list(MODEL_CONFIG[source]['prompt_styles'].keys())

    if request.method == 'POST' and prompt_key:
        # Derive target from prompt key (text after arrow)
        _, target = [p.strip() for p in prompt_key.split('→')]
        prefix = MODEL_CONFIG[source]['prompt_styles'][prompt_key]
        user_input = request.form.get('text_input', '')

        # Normalize input based on source and task type
        if source == 'Akkadian':
            if 'cuneiform' in prompt_key.lower():
                processed = normalizeString_cuneiform(
                    user_input,
                    use_prefix=False,
                    task="Translate",
                    language=source,
                    modern=target
                )
            else:
                processed = normalizeString_cuneiform_transliterate_translate(
                    user_input,
                    use_prefix=False,
                    task="Translate",
                    type="original",
                    language=source,
                    modern=target
                )
        else:
            processed = normalizeString_cuneiform_transliterate_translate(
                user_input,
                use_prefix=False,
                task="Translate",
                type="original",
                language=source,
                modern=target
            )

        # Tokenize and generate
        inputs = tokenizer(prefix + processed, return_tensors='pt')
        outputs = model.generate(**inputs, max_length=512)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return render_template(
        'index.html',
        translation=translation,
        sources=sources,
        selected_source=source,
        prompt_styles=prompt_styles,
        selected_prompt=prompt_key,
        prompt_map=prompt_map,
        request=request
    )

if __name__ == '__main__':
    app.run(debug=True)

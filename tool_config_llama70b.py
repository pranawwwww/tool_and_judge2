from codebase_rs import *

experiments = []

# ============================================================================
# Chinese and Hindi - all noise modes available
# ============================================================================
for noise in [AddNoiseMode.NoNoise, AddNoiseMode.Synonym, AddNoiseMode.Paraphrase]:
    for translate in [
        # English baseline (no translation)
        TranslateMode.NotTranslated(),
        
        # Chinese configurations
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.PartiallyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPromptTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPreTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPostTranslate),
        
        # Hindi configurations
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslated),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.PartiallyTranslated),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPromptTranslate),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPreTranslate),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPostTranslate),
    ]:
        experiments.append(ToolExperiment(translate, noise))

# ============================================================================
# Igbo - only NoNoise available (Synonym/Paraphrase datasets don't exist yet)
# NOTE: May need to rerun after verified Igbo version arrives
# ============================================================================
for translate in [
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.PartiallyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPromptTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPreTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPostTranslate),
]:
    experiments.append(ToolExperiment(translate, AddNoiseMode.NoNoise))

config = ToolConfig(
    Model.Local(LocalModel.Llama3_1_70B),
    experiments
)

from codebase_rs import *

experiments = []

# Chinese and Hindi - all noise modes
for noise in [AddNoiseMode.NoNoise, AddNoiseMode.Synonym, AddNoiseMode.Paraphrase]:
    for translate in [
        TranslateMode.NotTranslated(),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.PartiallyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPromptTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPreTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPostTranslate),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslated),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.PartiallyTranslated),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPromptTranslate),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPreTranslate),
        TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.FullyTranslatedPostTranslate),
    ]:
        experiments.append(ToolExperiment(translate, noise))

# Igbo - only NoNoise available
for translate in [
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.PartiallyTranslated),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPromptTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPreTranslate),
    TranslateMode.Translated(language=Language.Igbo, option=TranslateOption.FullyTranslatedPostTranslate),
]:
    experiments.append(ToolExperiment(translate, AddNoiseMode.NoNoise))

config = ToolConfig(
    Model.Local(LocalModel.Granite4_0HTiny),  # 0.3B parameters - very fast
    experiments
)


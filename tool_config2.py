from codebase_rs import *

experiments = []

# experiments.append(ToolExperiment(TranslateMode.NotTranslated(), AddNoiseMode.NoNoise))

for noise in [AddNoiseMode.NoNoise, AddNoiseMode.Synonym, AddNoiseMode.Paraphrase]:
    for translate in [
        TranslateMode.NotTranslated(),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.PartiallyTranslated),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPromptTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPreTranslate),
        TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPostTranslate),
    ]:
        experiments.append(ToolExperiment(translate, noise))

config = ToolConfig(
    Model.Api(ApiModel.Gpt5),
    experiments
)

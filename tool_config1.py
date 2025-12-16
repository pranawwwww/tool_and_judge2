from codebase_rs import *


configs = [
    # ToolConfig(Model.Api(ApiModel.Gpt5Nano), TranslateMode.NotTranslated(), AddNoiseMode.NoNoise),
    # ToolConfig(Model.Api(ApiModel.Gpt5Nano), TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPostTranslate), AddNoiseMode.Paraphrase),
    # ToolConfig(Model.Api(ApiModel.Gpt5Mini), TranslateMode.Translated(language=Language.Hindi, option=TranslateOption.PartiallyTranslated), AddNoiseMode.Synonym),
]


for model in [Model.Api(ApiModel.Gpt5Nano), Model.Api(ApiModel.Gpt5Mini), Model.Api(ApiModel.Gpt5)]:
    for noise in [AddNoiseMode.NoNoise, AddNoiseMode.Synonym, AddNoiseMode.Paraphrase]:
        for translate in [
            TranslateMode.NotTranslated(),
            TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslated),
            TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.PartiallyTranslated),
            TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPromptTranslate),
            TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPreTranslate),
            TranslateMode.Translated(language=Language.Chinese, option=TranslateOption.FullyTranslatedPostTranslate),
        ]:
            configs.append(ToolConfig(model, translate, noise))

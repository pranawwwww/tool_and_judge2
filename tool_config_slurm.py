from codebase_rs import *


configs = [
    # ToolConfig(Model.Local(LocalModel.Llama3_1_8B), TranslateMode.NotTranslated(), AddNoiseMode.NoNoise),
    
]


for model in [Model.Local(LocalModel.Llama3_1_8B)]:
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

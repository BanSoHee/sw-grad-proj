from googletrans import Translator

# googletrans
def get_translate(text, inlang, outlang):

    translator = Translator()
    trans = translator.translate(text, src=inlang, dest=outlang)

    print(f'translation result : {trans.text}')

    return trans.text

# Back Translation

# ko to en
def BT_ko2en(text):

    get_translate(text, 'ko', 'en')

    return

# en to ko
def BT_en2ko(text):

    get_translate(text, 'en', 'ko')

    return

# ko to jp
def BT_ko2jp(text):

    get_translate(text, 'ko', 'jp')

    return

# jp to ko
def BT_jp2ko(text):

    get_translate(text, 'jp', 'ko')

    return


''' sample '''
# get_translate('안녕하세요!', 'ko', 'en')
# BT_en2ko('hello')
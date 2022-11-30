from googletrans import Translator

# googletrans
def get_translate(text, inlang, outlang):

    translator = Translator()
    trans = translator.translate(text, src=inlang, dest=outlang)

    # print(f'translation result : {trans.text}')

    return trans.text

# Back Translation

# ko to en
def BT_ko2en(text):

    out = get_translate(text, 'ko', 'en')

    return out

# en to ko
def BT_en2ko(text):

    out = get_translate(text, 'en', 'ko')

    return out

# ko to jp
def BT_ko2jp(text):

    out = get_translate(text, 'ko', 'ja')

    return out

# jp to ko
def BT_jp2ko(text):

    out = get_translate(text, 'ja', 'ko')

    return out


''' sample '''
# get_translate('안녕하세요!', 'ko', 'en')
# out = BT_en2ko('hello!')
# print(out)
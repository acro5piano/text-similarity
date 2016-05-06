import os
import sys
import re
from gensim import corpora, matutils
import MeCab
import zenhan
import pandas as pd

DATA_PATH = './text/stretch_data.txt'
DICTIONARY_FILE_NAME = './dict/stretch_dic.txt'
mecab = MeCab.Tagger('mecabrc')

"""
TODO: Receive pandas Series and generate dict
      Now this script reads files.
"""


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    Extract alphabet as lower, hankaku, space-trimed
    '''

    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            try:
                yield zenhan.z2h(node.surface.lower().strip())
            except:
                yield '0'
        node = node.next


def check_stopwords(word):
    '''
    ストップワードだったらTrueを返す
    '''
    if re.search('^[0-9]+$', word):  # 数字だけ
        return True
    if re.search('^[0-9a-zA-Z\u3041-\u3093\u30A1-\u30F6\u4E00-\u9FD0]+$', word):  # White list [Alphabet, Kana, Kanji]
        return False
    return True


def get_words(contents):
    '''
    記事群のdictについて、形態素解析して返す
    '''
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    '''
    一つの記事を形態素解析して返す
    '''
    return [token for token in tokenize(content) if not check_stopwords(token)]


def filter_dictionary(dictionary):
    '''
    低頻度と高頻度のワードを除く感じで
    '''
    dictionary.filter_extremes(no_below=20, no_above=0.5)  # この数字はあとで変えるかも
    return dictionary


def get_contents():
    '''
    Extract stretch data
    '''

    df = pd.read_csv(DATA_PATH)
    return (df['KIBO_TANTOGYOMU_MEMO'] + df['KIBO_HOSPITALTYPE_MEMO'] + df['CONSCOMMENT'] + df['SHIGOTONAIYO']).to_dict()


def get_vector(dictionary, content):
    '''
    Analyze content and return a vector of feature using dictionary.
    @param  gensim_dict, str
    @return vector
    '''
    tmp = dictionary.doc2bow(get_words_main(content))
    dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
    return dense


def get_dictionary(create_flg=False, file_name=DICTIONARY_FILE_NAME):
    '''
    辞書を作る
    '''
    if create_flg or not os.path.exists(file_name):
        # データ読み込み
        contents = get_contents()
        # 形態素解析して名詞だけ取り出す
        words = get_words(contents)
        # 辞書作成、そのあとフィルタかける
        dictionary = filter_dictionary(corpora.Dictionary(words))
        # 保存しておく
        if file_name is None:
            sys.exit()
        dictionary.save_as_text(file_name)

    else:
        # 通常はファイルから読み込むだけにする
        dictionary = corpora.Dictionary.load_from_text(file_name)

    return dictionary


if __name__ == '__main__':
    get_dictionary(create_flg=True)


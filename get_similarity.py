# -*- coding: utf-8 -*-
from scipy.spatial.distance import cosine
import corpus

# 辞書の読み込み
dictionary = corpus.get_dictionary(create_flg=False)

def get_similarity(article1, article2):
    """Receive 2 articles (as str) then return their similarity"""

    # 記事の読み込み
    contents = corpus.get_contents()

    # 特徴抽出
    feature1 = corpus.get_vector(dictionary, article1)
    feature2 = corpus.get_vector(dictionary, article2)

    return cosine(feature1, feature2)

def main():
    article1 = corpus.get_file_content('text/dokujo-tsushin/dokujo-tsushin-4880091.txt')
    article2 = corpus.get_file_content('text/dokujo-tsushin/dokujo-tsushin-4880092.txt')
    print(get_similarity(article1, article2))

if __name__ == '__main__':
    main()

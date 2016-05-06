from scipy.spatial.distance import cosine
import pandas as pd
import corpus

def get_similarity(article1, article2):
    """Receive 2 articles (as str) then return their similarity"""

    # 辞書の読み込み
    dictionary = corpus.get_dictionary(create_flg=False)

    # 特徴抽出
    feature1 = corpus.get_vector(dictionary, article1)
    feature2 = corpus.get_vector(dictionary, article2)

    return cosine(feature1, feature2)

def main():
    df = pd.read_csv(corpus.DATA_PATH)

    for i in range(len(df)):
        memo1 = df['KIBO_TANTOGYOMU_MEMO'][i] + df['KIBO_HOSPITALTYPE_MEMO'][i]
        memo2 = df['CONSCOMMENT'][i] + df['SHIGOTONAIYO'][i]
        print('Similarity========\n{2}\nCareer========\n{0}\nOrder=========\n{1}\n\n'.format(memo1, memo2, get_similarity(memo1, memo2)))

if __name__ == '__main__':
    main()


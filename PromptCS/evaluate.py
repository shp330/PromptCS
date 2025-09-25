import argparse
import re
from bleu import bleuFromMaps
from sentence_transformers import SentenceTransformer, util
import math

def splitPuncts(line):
    """
    将输入字符串 line 中的单词和非空白的非单词字符（主要是标点符号）分割开，并用空格连接成一个新的字符串。
    """
    return ' '.join(re.findall(r"[\w]+|[^\s\w]", line))


def euclidean_distance(x, y):
    """
    欧几里得距离
    x: 多维向量 x
    y: 多维向量 y
    """
    if len(x) != len(y):
        raise ValueError("向量 x 和 y 的长度必须相同")
    return math.dist(x, y)


def sentence_bert_score_cos(output, gold):
    embeddings1 = model.encode(output, convert_to_tensor=True)
    embeddings2 = model.encode(gold, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    score = 0
    num = 0
    for i in range(len(output)):
        num = num + 1
        score = score + cosine_scores[i][i]

    print("Number of cases:{}".format(num))
    print("Score: {:.4f}".format(score/num))
    ans = (score/num).item()
    return round(ans, 4)

def sentence_bert_score(output, gold):
    print('**********')
    print('Sentence Bert + cosine similarity')
    value_cos = sentence_bert_score_cos(output, gold)
    print('**********')

    return value_cos


def bleu(output, gold):
    print('**********')
    print('Bleu:')
    score = 0
    num = 0
    for i in range(len(output)):
        num = num + 1
        predictionMap = {}
        goldMap = {}
        predictionMap[i] = [splitPuncts(output[i].strip().lower())]
        goldMap[i] = [splitPuncts(gold[i].strip().lower())]
        dev_bleu = round(bleuFromMaps(goldMap, predictionMap)[0], 2)
        score += dev_bleu

    print("Number of cases:{}".format(num))
    print("Score: {:.3f}".format(score/num))
    return round(score / num, 4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--predict_file_path", default='./saved_models/test_0.output', type=str)
    parser.add_argument("--ground_truth_file_path", default='./saved_models/test_0.gold', type=str)
    parser.add_argument("--SentenceBERT_model_path", default='../all-MiniLM-L6-v2', type=str)

    # print arguments
    args = parser.parse_args()

    model = SentenceTransformer(args.SentenceBERT_model_path)

    output = []
    gold = []
    i = 0
    with open(args.predict_file_path, "r", encoding="utf-8") as f:
        for line in f:
            comment = line.strip().split('\t')[-1]
            output.append(comment)

    with open(args.ground_truth_file_path, "r", encoding="utf-8") as f:
        for line in f:
            comment = line.strip().split('\t')[-1]
            gold.append(comment)

    sentence_bert_score(output, gold)
    bleu(output, gold)



